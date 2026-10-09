import tqdm
import torch
import numpy as np
from typing import List, Dict
from mine_utils import VectorizeProject, concat_joint
from mine_train import estimate_mi_with_mine

@torch.no_grad()
def _to_vec(proj: VectorizeProject, x: torch.Tensor) -> torch.Tensor:
    proj.eval()
    return proj(x)

def compute_layerwise_mi(
    acts_per_layer: List[torch.Tensor],  # list of [N,C,H,W] or [N,D]
    noise: torch.Tensor,                 # [N,Cn,Hn,Wn] or [N,Dn]
    target: torch.Tensor,                # [N,Ct,Ht,Wt] or [N,Dt]
    proj_dim: int = 256,
    train_steps: int = 800,
    device: str = "cuda",
    critic_hidden = 512,
) -> Dict[str, List[float]]:
    """
    Returns MI curves:
      - MI_AN[l] = I(A^l; N)
      - MI_AT[l] = I(A^l; T)
      - MI_ANT[l] = I(A^l; [N,T])
      - MI_AN_given_T[l] = I(A^l; N | T) = MI_ANT - MI_AT
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Projection heads (shared to keep things simple/consistent)
    proj_act = VectorizeProject(out_dim=proj_dim).to(dev)
    proj_noise = VectorizeProject(out_dim=proj_dim).to(dev)
    proj_target = VectorizeProject(out_dim=proj_dim).to(dev)

    # Project noise & target once
    n_vec = _to_vec(proj_noise, noise.to(dev))   # [N, d]
    t_vec = _to_vec(proj_target, target.to(dev)) # [N, d]
    nt_vec = concat_joint(n_vec, t_vec)          # [N, 2d]

    MI_AN, MI_AT, MI_ANT, MI_AN_given_T = [], [], [], []

    for idx, A in enumerate(tqdm.tqdm(acts_per_layer)):
        A = torch.stack([torch.from_numpy(np.load(a)).float() for a in A])
        a_vec = _to_vec(proj_act, A.to(dev))     # [N, d]

        mi_an  = estimate_mi_with_mine(a_vec, n_vec, steps=train_steps, device=dev, verbose=False, hidden=critic_hidden)
        mi_at  = estimate_mi_with_mine(a_vec, t_vec, steps=train_steps, device=dev, verbose=False, hidden=critic_hidden)
        mi_ant = estimate_mi_with_mine(a_vec, nt_vec, steps=train_steps, device=dev, verbose=False, hidden=critic_hidden)

        MI_AN.append(mi_an)
        MI_AT.append(mi_at)
        MI_ANT.append(mi_ant)
        MI_AN_given_T.append(mi_ant - mi_at)

        print(f"[Layer {idx:02d}] I(A;N)={mi_an:.3f}  I(A;T)={mi_at:.3f}  I(A;[N,T])={mi_ant:.3f}  I(A;N|T)={mi_ant - mi_at:.3f}")

    return {
        "MI_AN": MI_AN,
        "MI_AT": MI_AT,
        "MI_ANT": MI_ANT,
        "MI_AN_given_T": MI_AN_given_T,
    }
