import torch


class LazinessLayer(torch.nn.Module):
    """
    A single elementwise multiplication with one laziness parameter per
    channel. This is run through a sigmoid so that this is a real laziness parameter.

    In the original diffusion matrix, `P := 1/2 I + 1/2 W D^{-1}`,
    `P x = (1/2 I + 1/2 W D^{-1}) x`
    Here we are making the two 1/2 factors learnable.
    `P x = (laziness I + (1 - laziness) W D^{-1}) x`
        with `propagated = W D^{-1} x`
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.laziness_logit = torch.nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: torch.Tensor, propagated: torch.Tensor) -> torch.Tensor:
        laziness = torch.nn.functional.sigmoid(self.laziness_logit)
        laziness = laziness.unsqueeze(dim=1)

        assert x.shape == propagated.shape
        assert len(x.shape) in [2, 3]
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            laziness = laziness.unsqueeze(dim=0)
            laziness = laziness.repeat(batch_size, 1, 1)

        return laziness * x + (1 - laziness) * propagated

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.weights)
