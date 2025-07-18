import math
from torch.optim import Optimizer


class CosineWDSchedule(object):
    """
    Cosine decay weight decay scheduler.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        ref_wd: float,
        T_max: int,
        final_wd: float=0.
    ):
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to apply weight decay to.
        ref_wd : float
            The reference weight decay value to start from.
        final_wd : float, optional
            The final weight decay value to reach at the end
            of the schedule (default is 0.0).
        T_max : int
            The maximum number of steps for the scheduler.
        """
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        """
        Step the scheduler to update the weight decay value.
        """
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (
            self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
