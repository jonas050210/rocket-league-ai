"""Narrow upstream compatibility adapter: pageable CPU buffers and explicit GAE restore."""

from dataclasses import replace
from io import BytesIO

import torch
from rlgym_learn_algos.ppo import NumpyExperienceBuffer
from rlgym_learn_algos.util.circular_buffers import TensorCircularBuffer


class PortableExperienceBuffer(NumpyExperienceBuffer):
    def __init__(self, trajectory_processor, *, pin_memory: bool):
        super().__init__(trajectory_processor)
        self.use_pinned_memory = pin_memory

    def load(self, config):
        super().load(config)
        # Upstream hardcodes True. Allocation is lazy; change it before the first submit.
        for buffer in (self.log_probs, self.values, self.advantages):
            buffer.pin_memory = self.use_pinned_memory
        if config.checkpoint_load_folder:
            # Upstream 0.4.2 omits this checkpoint context when calling the GAE loader.
            self.trajectory_processor.load(replace(
                self.trajectory_processor.config, checkpoint_load_folder=config.checkpoint_load_folder,
            ))

    def _load_tensor_buffer_from_zip(self, archive, filename, convert_dtype=True):
        # Do not monkeypatch torch or upstream modules process-wide.
        buffer = TensorCircularBuffer(
            capacity=self.config.experience_buffer_config.max_size,
            device=self.config.experience_buffer_config.device,
            pin_memory=self.use_pinned_memory,
            forced_dtype=self.config.dtype if convert_dtype else None,
        )
        if filename in archive.namelist():
            state = torch.load(BytesIO(archive.read(filename)), map_location="cpu", weights_only=True)
            buffer.append(state["data"])
        return buffer
