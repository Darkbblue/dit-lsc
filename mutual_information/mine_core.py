import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

class MINECritic(nn.Module):
    """
    T_theta(x, y): simple MLP over concatenated (x||y) vectors.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))  # [B,1]

@torch.no_grad()
def _exp_moving_avg(prev, new, ema=0.99):
    if prev is None:
        return new
    return ema * prev + (1 - ema) * new

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None

def mine_dv_bound(critic, x, y, ma_et: torch.Tensor = None, ema=0.99):
    """
    Donsker–Varadhan lower bound with moving-average baseline (Belghazi et al., MINE).
    Returns: (mi_estimate, loss, new_ma_et)
    """
    # # Positive pairs
    # t_pos = critic(x, y)  # [B,1]
    # print('t_pos', t_pos.min(), t_pos.max())
    # # Negative pairs
    # y_neg = y[torch.randperm(y.size(0))]
    # t_neg = critic(x, y_neg)  # [B,1]
    # print('t_neg', t_neg.min(), t_neg.max())

    # # DV bound: E[T] - log E[e^T]  (use moving average for the second term)
    # et = torch.exp(t_neg).mean().detach()
    # ma_et = _exp_moving_avg(ma_et, et, ema=ema)
    # print('ma_et', ma_et)

    # # Bias-reduced objective
    # loss = -(t_pos.mean() - torch.log(ma_et + 1e-8))
    # print('loss', loss)
    # mi = t_pos.mean() - torch.log(torch.exp(t_neg).mean() + 1e-8)
    # return mi.detach(), loss, ma_et

    y_marg = y[torch.randperm(x.shape[0])]

    t = critic(x, y).mean()
    t_marg = critic(x, y_marg)

    def ema_loss(x, running_mean, alpha):
        def compute(mu, alpha, past_ema):
            return alpha * mu + (1.0 - alpha) * past_ema
        t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
        if running_mean == 0:
            running_mean = t_exp
        else:
            running_mean = compute(t_exp, alpha, running_mean.item())
        t_log = EMALoss.apply(x, running_mean)

        # Recalculate ema

        return t_log, running_mean

    second_term, ma_et = ema_loss(
        t_marg, ma_et, ema
    )

    loss = -t + second_term
    mi = -loss
    return mi, loss, ma_et
