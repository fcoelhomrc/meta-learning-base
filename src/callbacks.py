from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional
import torch
from pathlib import Path
import os
import warnings

class Callback(ABC):
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    @abstractmethod
    def trigger(self, **trigger_kwargs):
        ...

    @abstractmethod
    def check(self, **check_kwargs) -> bool:
        ...

    def _eval(self, trigger_kwargs, check_kwargs):
        if self.check(**check_kwargs):
            self.trigger(**trigger_kwargs)



class ModelCheckpoint(Callback):
    EXT = ".ckpt"
    def __init__(self, save_name: str, save_path: str, mode: str = "min", logger: Optional[Logger] = None):
        super().__init__(logger)
        self.save_name = save_name
        self.save_path = save_path
        self.mode = mode
        self.score = None
        assert mode in ["min", "max"]

    def trigger(self, module: torch.nn.Module):
        target = os.path.join(self.save_path, self.save_name + ModelCheckpoint.EXT)
        if os.path.exists(target):
            warnings.warn(f"Checkpoint {target} exists and will be overwritten.")
        torch.save(module.state_dict(), target)

    def check(self, score: float) -> bool:
        if self.score is None:
            self.score = score
            return True
        should_trigger = score < self.score if self.mode == "min" else score > self.score
        if should_trigger:
            self.score = score
        return should_trigger

    def eval(self, module: torch.nn.Module, score: float) -> None:
        super()._eval(
            trigger_kwargs={"module": module},
            check_kwargs={"score": score},
        )