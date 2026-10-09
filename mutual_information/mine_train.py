import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from mine_core import MINECritic, mine_dv_bound
# from mine.models.mine import Mine

def estimate_mi_with_mine(
    x_vec: torch.Tensor,           # [N, Dx] after projection
    y_vec: torch.Tensor,           # [N, Dy] after projection
    steps: int = 800,
    batch_size: int = 512,
    hidden: int = 512,
    lr: float = 1e-4,
    ema: float = 0.99,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> float:
    """
    Trains a MINE critic on (x_vec, y_vec) to estimate I(X;Y).
    Returns the final MI estimate (scalar).
    """
    assert x_vec.shape[0] == y_vec.shape[0], "x and y must have same N"
    N, Dx = x_vec.shape
    Dy = y_vec.shape[1]

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_vec, y_vec = x_vec.to(device), y_vec.to(device)

    critic = MINECritic(Dx, Dy, hidden=hidden).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=lr)
    ds = TensorDataset(x_vec, y_vec)
    dl = DataLoader(ds, batch_size=min(batch_size, N), shuffle=True)

    ma_et = 0
    mi_running = None

    step = 0
    while step < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            mi, loss, ma_et = mine_dv_bound(critic, xb, yb, ma_et=ma_et, ema=ema)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()

            mi_running = mi if mi_running is None else 0.9 * mi_running + 0.1 * mi
            step += 1
            if verbose and step % 100 == 0:
                print(f"[MINE] step {step:04d}  MI≈{mi_running.item():.4f}")
            if step >= steps:
                break

    return float(mi_running.item() if mi_running is not None else mi.item())

    # assert x_vec.shape[0] == y_vec.shape[0], "x and y must have same N"
    # N, Dx = x_vec.shape
    # Dy = y_vec.shape[1]

    # device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # x_vec, y_vec = x_vec.to(device), y_vec.to(device)

    # critic = MINECritic(Dx, Dy, hidden=hidden).to(device)

    # mine = Mine(
    #     T = critic,
    #     loss = 'mine',
    #     method = 'concat'
    # )

    # mi = mine.optimize(x_vec, y_vec, iters = 100)
    # return mi
