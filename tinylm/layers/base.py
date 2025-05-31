from torch import nn

from tinylm.config import get_config


class AcceleratedModule[T](nn.Module):
    def __init__(self) -> None:
        super().__init__()

        config = get_config()
        if config.use_flashinfer:
            self.forward = self.forward_flashinfer
        else:
            self.forward = self.forward_torch

    def forward_flashinfer(self, *args, **kwargs) -> T:
        raise NotImplementedError("no operation defined for flashinfer")

    def forward_torch(self, *args, **kwargs) -> T:
        raise NotImplementedError("no operation defined for torch")
