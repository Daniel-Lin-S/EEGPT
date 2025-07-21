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


class WarmupCosineSchedule(object):
    """
    Warmup and cosine decay learning rate scheduler.
    This scheduler first linearly increases the learning rate
    from a starting value to a reference value over a specified
    number of warmup steps, and then applies a cosine decay
    to the learning rate for the remaining steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_lr: float,
        ref_lr: float,
        T_max: int,
        final_lr: float=0.
    ):
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to apply the learning rate schedule to.
        warmup_steps : int
            The number of steps for the warmup phase.
        start_lr : float
            The initial learning rate at the start of the warmup phase.
        ref_lr : float
            The reference learning rate to reach at
            the end of the warmup phase.
        T_max : int
            The total number of steps for the cosine decay phase,
            excluding the warmup steps.
        final_lr : float, optional
            The final learning rate to reach at the end of the schedule
            (default is 0.0).
        """
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self) -> float:
        """
        Step the scheduler to update the learning rate.
    
        Return
        ------
        float
            The new learning rate after the step.
        """
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (
                    1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr
