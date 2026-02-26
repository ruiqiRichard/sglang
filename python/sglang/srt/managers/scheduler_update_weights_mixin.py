from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:
    def _merge_release_memory_occupation_req(
        self: Scheduler,
        recv_req: ReleaseMemoryOccupationReqInput,
    ):
        pending_req = self.pending_release_memory_occupation_req
        if pending_req is None:
            self.pending_release_memory_occupation_req = recv_req
            return

        # None/[] means "all tags", which dominates any narrower request.
        if not pending_req.tags or not recv_req.tags:
            pending_req.tags = None
            return

        merged = []
        seen = set()
        for tag in [*pending_req.tags, *recv_req.tags]:
            if tag not in seen:
                seen.add(tag)
                merged.append(tag)
        pending_req.tags = merged

    def _release_memory_occupation_impl(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self._is_no_request()
        ), "release_memory_occupation should be called only when no ongoing request."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.cuda.synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def try_complete_pending_release_memory_occupation(self: Scheduler):
        pending_req = self.pending_release_memory_occupation_req
        if pending_req is None or not self._is_no_request():
            return

        logger.info("Scheduler became idle; proceeding with deferred release_memory_occupation")
        self.pending_release_memory_occupation_req = None
        output = self._release_memory_occupation_impl(pending_req)
        self.offload_transitioning = False
        if self.recv_from_rpc is not None:
            self.recv_from_rpc.send_pyobj(output)

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            flush_cache_success = self.flush_cache()
            assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        worker = self.draft_worker if self.server_args.speculative_algorithm == "STANDALONE_OPD" else self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        self.offload_transitioning = True
        if not self._is_no_request():
            logger.info(
                "Deferring release_memory_occupation until scheduler is idle "
                "(waiting=%d, running_empty=%s, last_empty=%s, cur_empty=%s)",
                len(self.waiting_queue),
                self.running_batch.is_empty(),
                self.last_batch is None or self.last_batch.is_empty(),
                self.cur_batch is None or self.cur_batch.is_empty(),
            )
            self._merge_release_memory_occupation_req(recv_req)
            return None

        try:
            return self._release_memory_occupation_impl(recv_req)
        finally:
            self.offload_transitioning = False

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        assert (
            self.pending_release_memory_occupation_req is None
        ), "resume_memory_occupation should not be called while release is pending."
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor
