# src/lr_schedules.py
from __future__ import annotations

class PaperStyleLR:
    def __init__(self, optimizer, *,
                 gamma: float = 0.998,
                 bump: float = 1.008,
                 patience: int = 4,
                 tol: float = 5e-5,
                 min_lr: float = 1e-6,
                 max_lr: float = 3e-3,
                 bump_hold: int = 3,
                 max_bumps: int = 64,
                 ema_beta: float = 0.9,
                 total_epochs: int | None = None,
                 cutoff_frac: float | None = None,
                 after_cutoff_gamma: float | None = None):
        """
        Adaptive LR schedule inspired by the 'paper-style' schedule:
        - Decays by gamma each epoch.
        - If val loss stagnates for 'patience' epochs, bumps LR up by 'bump' factor.
        - After a bump, holds for 'bump_hold' epochs.
        - Supports a second decay regime after cutoff_frac * total_epochs.
        """

        self.opt = optimizer
        self.gamma = gamma
        self.bump = bump
        self.patience = max(1, patience)
        self.tol = tol
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.bump_hold = int(max(0, bump_hold))
        self.max_bumps = int(max_bumps)
        self.ema_beta = float(ema_beta)

        self.total_epochs = total_epochs
        self.cutoff_frac = cutoff_frac
        self.after_cutoff_gamma = after_cutoff_gamma

        # Tracking
        self.best = float("inf")
        self.bad_epochs = 0
        self.bumps_done = 0
        self.hold_left = 0
        self.epoch = 0
        self.ema_val = None

        for g in self.opt.param_groups:
            g["lr"] = max(self.min_lr, min(self.max_lr, g["lr"]))

    def _set_lr(self, new_lr: float):
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        for g in self.opt.param_groups:
            g["lr"] = new_lr
        return new_lr

    def _get_lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    def step(self, val: float) -> float:
        self.epoch += 1

        # Exponential moving average for smooth val tracking
        self.ema_val = val if self.ema_val is None else (
            self.ema_beta * self.ema_val + (1 - self.ema_beta) * val
        )

        # Detect improvement
        if self.ema_val < self.best - self.tol:
            self.best = self.ema_val
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        lr_now = self._get_lr()

        # --- Bump phase ---
        if self.hold_left == 0 and self.bad_epochs >= self.patience and self.bumps_done < self.max_bumps:
            new_lr = min(lr_now * self.bump, self.max_lr)
            if new_lr > lr_now * (1 + 1e-12):
                lr_now = self._set_lr(new_lr)
                self.bumps_done += 1
                self.bad_epochs = 0
                self.hold_left = self.bump_hold

        # --- Decay phase ---
        if self.hold_left > 0:
            self.hold_left -= 1
        else:
            gamma = self.gamma
            if (
                self.total_epochs is not None
                and self.cutoff_frac is not None
                and self.after_cutoff_gamma is not None
            ):
                cutoff_ep = max(1, int(round(self.cutoff_frac * self.total_epochs)))
                if self.epoch >= cutoff_ep:
                    gamma = self.after_cutoff_gamma
            lr_now = self._set_lr(lr_now * gamma)

        return lr_now
