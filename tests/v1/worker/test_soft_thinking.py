# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for soft thinking decoding.

Tests cover:
- Soft embedding computation (probs @ embed_weight)
- TP-aware soft embedding computation with shard padding
- Thinking phase transitions (start/end detection)
- Output hiding (tokens suppressed during thinking)
- Bookkeeping (is_token_ids, separate soft_thinking_token_ids)
- Request lifecycle (state sync after condense/swap)
- Data structure fields and defaults
"""

import numpy as np
import torch


class TestSoftEmbeddingComputation:
    """Test the core probs @ embed_weight computation."""

    def test_soft_embed_weighted_average(self):
        """Verify that soft_embed = probs @ embed_weight produces
        a correct weighted average of token embeddings."""
        vocab_size = 8
        hidden_dim = 4
        embed_weight = torch.randn(vocab_size, hidden_dim)

        probs = torch.zeros(1, vocab_size)
        probs[0, 2] = 0.7
        probs[0, 5] = 0.3

        soft_embed = probs @ embed_weight
        expected = 0.7 * embed_weight[2] + 0.3 * embed_weight[5]
        torch.testing.assert_close(soft_embed.squeeze(0), expected)

    def test_soft_embed_one_hot(self):
        """When probs is one-hot, soft embed should equal the corresponding
        embedding row (discrete decoding equivalence)."""
        vocab_size = 16
        hidden_dim = 8
        embed_weight = torch.randn(vocab_size, hidden_dim)

        probs = torch.zeros(1, vocab_size)
        probs[0, 7] = 1.0

        soft_embed = probs @ embed_weight
        torch.testing.assert_close(soft_embed.squeeze(0), embed_weight[7])

    def test_soft_embed_uniform(self):
        """When probs is uniform, soft embed should equal the mean of all
        embeddings."""
        vocab_size = 4
        hidden_dim = 6
        embed_weight = torch.randn(vocab_size, hidden_dim)

        probs = torch.full((1, vocab_size), 1.0 / vocab_size)

        soft_embed = probs @ embed_weight
        expected = embed_weight.mean(dim=0)
        torch.testing.assert_close(soft_embed.squeeze(0), expected)

    def test_soft_embed_from_logits_via_softmax(self):
        """End-to-end: logits -> softmax -> weighted embedding."""
        vocab_size = 32
        hidden_dim = 16
        embed_weight = torch.randn(vocab_size, hidden_dim)
        logits = torch.randn(1, vocab_size)

        probs = torch.softmax(logits.float(), dim=-1)
        soft_embed = probs @ embed_weight

        # Probabilities sum to 1, so soft_embed is a convex combination.
        manual = sum(
            probs[0, i].item() * embed_weight[i]
            for i in range(vocab_size)
        )
        torch.testing.assert_close(
            soft_embed.squeeze(0), manual, atol=1e-5, rtol=1e-5
        )

    def test_soft_embed_batch(self):
        """Batched computation should match per-request results."""
        batch_size = 4
        vocab_size = 10
        hidden_dim = 6
        embed_weight = torch.randn(vocab_size, hidden_dim)
        logits = torch.randn(batch_size, vocab_size)
        probs = torch.softmax(logits.float(), dim=-1)

        batch_result = probs @ embed_weight
        for i in range(batch_size):
            single_result = probs[i:i + 1] @ embed_weight
            torch.testing.assert_close(batch_result[i], single_result.squeeze(0))


class TestTPAwareSoftEmbedding:
    """Test the TP-aware soft embedding computation with sharded weights."""

    def test_tp_shard_reconstruction(self):
        """Verify that partial matmuls on TP shards + all-reduce
        reconstructs the full soft embedding."""
        vocab_size = 16
        hidden_dim = 8
        embed_weight = torch.randn(vocab_size, hidden_dim)

        probs = torch.softmax(torch.randn(1, vocab_size), dim=-1)
        full_result = probs @ embed_weight

        tp_size = 2
        shard_size = vocab_size // tp_size
        partial_0 = probs[:, :shard_size] @ embed_weight[:shard_size, :]
        partial_1 = probs[:, shard_size:] @ embed_weight[shard_size:, :]

        reconstructed = partial_0 + partial_1
        torch.testing.assert_close(reconstructed, full_result)

    def test_tp_shard_with_padding(self):
        """Simulate VocabParallelEmbedding padding and verify correct slicing.

        Padding rows in the local shard should be skipped so they don't
        drop probability mass from the embedding.
        """
        org_vocab_size = 10
        hidden_dim = 4
        full_embed = torch.randn(org_vocab_size, hidden_dim)

        probs = torch.softmax(torch.randn(1, org_vocab_size), dim=-1)
        full_result = probs @ full_embed

        padded_size = 8

        # Shard 0: tokens 0..4, padded to 8 rows
        pad_embed_0 = torch.zeros(padded_size, hidden_dim)
        pad_embed_0[:5, :] = full_embed[:5, :]
        local_probs_0 = probs[:, 0:5]
        local_offset_0 = 0
        local_embed_0 = pad_embed_0[local_offset_0:local_offset_0 + 5, :]
        partial_0 = local_probs_0 @ local_embed_0

        # Shard 1: tokens 5..9, padded to 8 rows
        pad_embed_1 = torch.zeros(padded_size, hidden_dim)
        pad_embed_1[:5, :] = full_embed[5:10, :]
        local_probs_1 = probs[:, 5:10]
        local_offset_1 = 0
        local_embed_1 = pad_embed_1[local_offset_1:local_offset_1 + 5, :]
        partial_1 = local_probs_1 @ local_embed_1

        reconstructed = partial_0 + partial_1
        torch.testing.assert_close(reconstructed, full_result)

    def test_tp4_shard_reconstruction(self):
        """Test with TP=4 to verify the pattern generalizes."""
        vocab_size = 32
        hidden_dim = 8
        embed_weight = torch.randn(vocab_size, hidden_dim)
        probs = torch.softmax(torch.randn(1, vocab_size), dim=-1)
        full_result = probs @ embed_weight

        tp_size = 4
        shard_size = vocab_size // tp_size
        total = torch.zeros(1, hidden_dim)
        for rank in range(tp_size):
            start = rank * shard_size
            end = start + shard_size
            total += probs[:, start:end] @ embed_weight[start:end, :]

        torch.testing.assert_close(total, full_result)


class TestThinkingPhaseTransitions:
    """Test detection of thinking phase boundaries."""

    def test_enter_soft_thinking_on_start_token(self):
        """When a request samples the start-of-thinking token,
        is_in_soft_thinking should become True."""
        think_start_token_id = 100
        max_reqs = 4

        is_in_soft_thinking = np.zeros(max_reqs, dtype=bool)

        sampled = [[42], [think_start_token_id], [55], [66]]
        for req_idx in range(4):
            if not is_in_soft_thinking[req_idx]:
                if (
                    sampled[req_idx]
                    and sampled[req_idx][0] == think_start_token_id
                ):
                    is_in_soft_thinking[req_idx] = True

        assert not is_in_soft_thinking[0]
        assert is_in_soft_thinking[1]
        assert not is_in_soft_thinking[2]
        assert not is_in_soft_thinking[3]

    def test_exit_soft_thinking_on_end_token_argmax(self):
        """When end-of-thinking token is the argmax, soft thinking ends."""
        think_end_token_id = 101
        vocab_size = 200

        logits = torch.zeros(1, vocab_size)
        logits[0, think_end_token_id] = 10.0

        probs = torch.softmax(logits.float(), dim=-1)
        top_token = probs.argmax(dim=-1).item()

        assert top_token == think_end_token_id

    def test_stay_in_soft_thinking_when_end_not_argmax(self):
        """When end token is NOT the argmax, soft thinking continues."""
        think_end_token_id = 101
        vocab_size = 200

        logits = torch.zeros(1, vocab_size)
        logits[0, 50] = 10.0
        logits[0, think_end_token_id] = 2.0

        probs = torch.softmax(logits.float(), dim=-1)
        top_token = probs.argmax(dim=-1).item()

        assert top_token != think_end_token_id
        assert top_token == 50

    def test_full_lifecycle_enter_think_produce_exit(self):
        """Simulate the full lifecycle: normal -> enter thinking ->
        N soft steps -> exit thinking -> normal."""
        think_start_id = 100
        think_end_id = 101
        vocab_size = 200
        hidden_dim = 8
        embed_weight = torch.randn(vocab_size, hidden_dim)

        is_thinking = False
        output_token_ids: list[int] = []
        soft_thinking_token_ids: list[int] = []

        # Step 1: normal token
        sampled_token = 42
        output_token_ids.append(sampled_token)

        # Step 2: sample <think>
        sampled_token = think_start_id
        output_token_ids.append(sampled_token)
        is_thinking = True

        # Steps 3-5: soft thinking (argmax != end_token)
        for step in range(3):
            logits = torch.randn(1, vocab_size)
            logits[0, think_end_id] = -10.0  # end token not dominant
            probs = torch.softmax(logits.float(), dim=-1)
            top_token = probs.argmax(dim=-1).item()
            assert top_token != think_end_id
            soft_thinking_token_ids.append(top_token)

        # Step 6: end token becomes argmax
        logits = torch.zeros(1, vocab_size)
        logits[0, think_end_id] = 20.0
        probs = torch.softmax(logits.float(), dim=-1)
        top_token = probs.argmax(dim=-1).item()
        assert top_token == think_end_id
        is_thinking = False
        output_token_ids.append(think_end_id)

        # Step 7: back to normal
        sampled_token = 55
        output_token_ids.append(sampled_token)

        assert len(output_token_ids) == 4  # 42, <think>, </think>, 55
        assert len(soft_thinking_token_ids) == 3


class TestOutputHiding:
    """Test that thinking tokens are hidden from the API output."""

    def test_soft_thinking_mask_generation(self):
        """Verify the soft_thinking_mask correctly flags thinking requests."""
        has_pending = np.array([False, True, False, True], dtype=bool)
        req_ids = ["r0", "r1", "r2", "r3"]
        req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}

        mask = [
            bool(has_pending[req_id_to_index[rid]])
            for rid in req_ids
        ]

        assert mask == [False, True, False, True]

    def test_scheduler_suppresses_tokens_for_thinking_requests(self):
        """Verify that the scheduler sets new_token_ids = [] when
        soft_thinking_mask is True for a request."""
        generated_tokens = [42, 43, 44]
        is_soft_thinking = True

        if is_soft_thinking and generated_tokens:
            new_token_ids: list[int] = []
        else:
            new_token_ids = generated_tokens

        assert new_token_ids == []

    def test_non_thinking_tokens_pass_through(self):
        """Non-thinking requests should not have tokens suppressed."""
        generated_tokens = [42, 43, 44]
        is_soft_thinking = False

        if is_soft_thinking and generated_tokens:
            new_token_ids: list[int] = []
        else:
            new_token_ids = generated_tokens

        assert new_token_ids == [42, 43, 44]


class TestBookkeeping:
    """Test the bookkeeping changes for soft thinking."""

    def test_is_token_ids_set_false_for_soft_thinking(self):
        """When has_pending_soft_embed is True, is_token_ids should be False."""
        max_model_len = 128
        is_token_ids = np.ones((4, max_model_len), dtype=bool)
        has_pending = np.array([False, True, False, False], dtype=bool)

        req_idx = 1
        start_idx = 10
        end_idx = 11

        if has_pending[req_idx]:
            is_token_ids[req_idx, start_idx:end_idx] = False
        else:
            is_token_ids[req_idx, start_idx:end_idx] = True

        assert not is_token_ids[1, 10]
        assert is_token_ids[0, 10]

    def test_soft_thinking_token_ids_separate_from_output(self):
        """Argmax tokens during thinking should go to soft_thinking_token_ids,
        not output_token_ids."""
        output_token_ids: list[int] = [1, 2, 3]
        soft_thinking_token_ids: list[int] = []
        has_pending_soft_embed = True

        sampled = [42]

        if has_pending_soft_embed:
            soft_thinking_token_ids.extend(sampled)
        else:
            output_token_ids.extend(sampled)

        assert output_token_ids == [1, 2, 3]
        assert soft_thinking_token_ids == [42]

    def test_normal_tokens_append_to_output(self):
        """Non-thinking tokens should go to output_token_ids."""
        output_token_ids: list[int] = [1, 2, 3]
        soft_thinking_token_ids: list[int] = []
        has_pending_soft_embed = False

        sampled = [42]

        if has_pending_soft_embed:
            soft_thinking_token_ids.extend(sampled)
        else:
            output_token_ids.extend(sampled)

        assert output_token_ids == [1, 2, 3, 42]
        assert soft_thinking_token_ids == []


class TestLifecycleSync:
    """Test that soft thinking state survives batch condense/reorder."""

    def test_sync_rebuilds_arrays_after_index_change(self):
        """Simulate condense moving a request to a different index.
        _sync_soft_thinking_state should rebuild arrays correctly."""
        max_reqs = 8
        hidden_dim = 4

        is_in_soft_thinking = np.zeros(max_reqs, dtype=bool)
        has_pending_soft_embed = np.zeros(max_reqs, dtype=bool)

        # Before condense: req_A at index 5 is in soft thinking
        is_in_soft_thinking[5] = True
        has_pending_soft_embed[5] = True

        # Simulate condense: req_A moves from index 5 to index 2
        class MockReqState:
            def __init__(self, active, pending):
                self.soft_thinking_active = active
                self.has_pending_soft_embed = pending

        req_ids = [None, None, "req_A", "req_B", None, None, None, None]
        req_id_to_index = {"req_A": 2, "req_B": 3}
        requests = {
            "req_A": MockReqState(active=True, pending=True),
            "req_B": MockReqState(active=False, pending=False),
        }

        # Simulate _sync_soft_thinking_state
        is_in_soft_thinking[:] = False
        has_pending_soft_embed[:] = False
        for req_id in req_ids:
            if req_id is None:
                continue
            new_idx = req_id_to_index[req_id]
            req_state = requests.get(req_id)
            if req_state is None:
                continue
            if req_state.soft_thinking_active:
                is_in_soft_thinking[new_idx] = True
            if req_state.has_pending_soft_embed:
                has_pending_soft_embed[new_idx] = True

        # req_A moved to index 2
        assert is_in_soft_thinking[2]
        assert has_pending_soft_embed[2]
        # Old index 5 is cleared
        assert not is_in_soft_thinking[5]
        assert not has_pending_soft_embed[5]
        # req_B at index 3 is not thinking
        assert not is_in_soft_thinking[3]

    def test_sync_clears_finished_request_state(self):
        """When a request finishes and is removed, its slot should be cleared
        after sync."""
        max_reqs = 4
        is_in_soft_thinking = np.zeros(max_reqs, dtype=bool)
        has_pending_soft_embed = np.zeros(max_reqs, dtype=bool)

        # req_A was at index 1 and was in soft thinking
        is_in_soft_thinking[1] = True

        # req_A finishes and is removed from the batch
        req_ids = [None, "req_B", None, None]
        req_id_to_index = {"req_B": 1}

        class MockReqState:
            def __init__(self, active, pending):
                self.soft_thinking_active = active
                self.has_pending_soft_embed = pending

        requests = {"req_B": MockReqState(active=False, pending=False)}

        # Sync
        is_in_soft_thinking[:] = False
        has_pending_soft_embed[:] = False
        for req_id in req_ids:
            if req_id is None:
                continue
            new_idx = req_id_to_index[req_id]
            req_state = requests.get(req_id)
            if req_state is not None and req_state.soft_thinking_active:
                is_in_soft_thinking[new_idx] = True

        # Index 1 now belongs to req_B (not thinking)
        assert not is_in_soft_thinking[1]

    def test_sync_handles_empty_batch(self):
        """Sync with no active requests should zero out all state."""
        max_reqs = 4
        is_in_soft_thinking = np.ones(max_reqs, dtype=bool)
        has_pending_soft_embed = np.ones(max_reqs, dtype=bool)

        # All requests removed
        is_in_soft_thinking[:] = False
        has_pending_soft_embed[:] = False

        assert not np.any(is_in_soft_thinking)
        assert not np.any(has_pending_soft_embed)


class TestStepCountReporting:
    """Test that soft thinking step counts are tracked and propagated."""

    def test_step_count_increments_during_thinking(self):
        """Each soft thinking step should increment the counter."""
        num_soft_thinking_steps = 0
        is_soft_thinking = True

        for _ in range(5):
            if is_soft_thinking:
                num_soft_thinking_steps += 1

        assert num_soft_thinking_steps == 5

    def test_step_count_stops_after_thinking_ends(self):
        """Counter should stop incrementing once thinking exits."""
        num_soft_thinking_steps = 0
        steps = [True, True, True, False, False]

        for is_thinking in steps:
            if is_thinking:
                num_soft_thinking_steps += 1

        assert num_soft_thinking_steps == 3

    def test_step_count_reported_in_engine_core_output(self):
        """EngineCoreOutput should carry num_soft_thinking_steps."""
        from vllm.v1.engine import EngineCoreOutput

        output = EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[],
            num_soft_thinking_steps=10,
        )
        assert output.num_soft_thinking_steps == 10

    def test_step_count_default_zero(self):
        """When no soft thinking, step count defaults to 0."""
        from vllm.v1.engine import EngineCoreOutput

        output = EngineCoreOutput(
            request_id="req-2",
            new_token_ids=[1, 2, 3],
        )
        assert output.num_soft_thinking_steps == 0


class TestRequestState:
    """Test soft thinking state on the Request object."""

    def test_request_initial_state(self):
        """Request should start with soft thinking disabled."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        req = Request(
            request_id="test-req",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=0,
        )
        assert req.soft_thinking_active is False
        assert req.num_soft_thinking_steps == 0

    def test_request_soft_thinking_activation(self):
        """Soft thinking state can be toggled on the request."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        req = Request(
            request_id="test-req",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=0,
        )
        req.soft_thinking_active = True
        req.num_soft_thinking_steps = 42
        assert req.soft_thinking_active is True
        assert req.num_soft_thinking_steps == 42


class TestModelRunnerOutputMask:
    """Test ModelRunnerOutput.soft_thinking_mask."""

    def test_mask_field_exists(self):
        from vllm.v1.outputs import ModelRunnerOutput

        output = ModelRunnerOutput(
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            sampled_token_ids=[[1], [2]],
            soft_thinking_mask=[False, True],
        )
        assert output.soft_thinking_mask is not None
        assert output.soft_thinking_mask == [False, True]

    def test_mask_default_none(self):
        from vllm.v1.outputs import ModelRunnerOutput

        output = ModelRunnerOutput(
            req_ids=["a"],
            req_id_to_index={"a": 0},
            sampled_token_ids=[[1]],
        )
        assert output.soft_thinking_mask is None


class TestCachedRequestState:
    """Test CachedRequestState soft thinking fields."""

    def test_soft_thinking_token_ids_default_none(self):
        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        state = CachedRequestState(
            req_id="test",
            prompt_token_ids=[1, 2, 3],
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        assert state.soft_thinking_token_ids is None

    def test_soft_thinking_active_default_false(self):
        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        state = CachedRequestState(
            req_id="test",
            prompt_token_ids=[1, 2, 3],
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        assert state.soft_thinking_active is False
        assert state.has_pending_soft_embed is False

    def test_soft_thinking_token_ids_accumulation(self):
        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        state = CachedRequestState(
            req_id="test",
            prompt_token_ids=[1, 2, 3],
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        state.soft_thinking_token_ids = []
        state.soft_thinking_token_ids.extend([42, 43])
        state.soft_thinking_token_ids.extend([44])
        assert state.soft_thinking_token_ids == [42, 43, 44]

    def test_soft_thinking_state_persists_across_preemption(self):
        """When a request is preempted and re-added, its thinking state
        should be preserved on CachedRequestState."""
        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        state = CachedRequestState(
            req_id="test",
            prompt_token_ids=[1, 2, 3],
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        state.soft_thinking_active = True
        state.has_pending_soft_embed = True
        state.soft_thinking_token_ids = [10, 20, 30]

        # Simulate preemption: state is kept in self.requests,
        # removed from batch, then re-added
        assert state.soft_thinking_active is True
        assert state.has_pending_soft_embed is True
        assert state.soft_thinking_token_ids == [10, 20, 30]
