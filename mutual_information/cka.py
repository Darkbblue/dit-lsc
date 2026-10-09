import os
import json
import tqdm
import glob
import torch
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=4000)
parser.add_argument('--feat_root', type=str)
parser.add_argument('--img_root', type=str)
parser.add_argument('--layers', type=str)

args = parser.parse_args()

N = args.N
feat_root = args.feat_root
img_root = args.img_root
layers = args.layers

# loading
with open(layers, 'r') as f:
    layers = json.load(f)
new_layers = []
for layer in layers.keys():
    if 'pipe' in layer:
        continue
    new_layers.append(layer)
layers = new_layers

all_samples = sorted(glob.glob(feat_root))[:N]

acts_per_layer = [[] for _ in range(len(layers))]
noise = []
target = []
print('loading samples')
for sample in tqdm.tqdm(all_samples):
    sample_name = sample.split('/')[-1]
    img = os.path.join(img_root, f'{sample_name}.png')

    noise_sample = torch.from_numpy(np.load(os.path.join(sample, 'pipe-gt.npy'))).float()
    noise.append(noise_sample)

    img = Image.open(img).convert('RGB')
    img = torch.from_numpy(np.array(img)).float()
    target.append(img)

    for i, layer in enumerate(layers):
        # feat = torch.from_numpy(np.load(os.path.join(sample, f'{layer}.npy'))).float()
        feat = os.path.join(sample, f'{layer}.npy')
        acts_per_layer[i].append(feat)

# for i in range(len(acts_per_layer)):
#     acts_per_layer[i] = torch.stack(acts_per_layer[i])
noise = torch.stack(noise)
target = torch.stack(target)

def linear_cka_2d(A, B, eps=1e-12):
    """
    A: [H, W, C]
    B: [H, W, Cb]
    return: scalar CKA in [0,1] (数值上可能有微小越界)
    """
    Ca, H, W = A.shape
    Cb, Hb, Wb = B.shape
    # assert H == Hb and W == Wb, "需要空间分辨率一致（或你先做对齐/resize）"
    if H != Hb or W != Wb:
        resize_target = (H, W)
        B = torch.nn.functional.interpolate(
            B.unsqueeze(0), resize_target
        ).squeeze(0)

    X = A.reshape(H*W, Ca).float()
    Y = B.reshape(H*W, Cb).float()

    # 中心化（对样本维 n=H*W）
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Frobenius norms
    XT_Y = X.t() @ Y
    num = (XT_Y**2).sum()

    XT_X = X.t() @ X
    YT_Y = Y.t() @ Y
    den = torch.sqrt((XT_X**2).sum() * (YT_Y**2).sum()).clamp_min(eps)

    return (num / den).item()

for layer_idx in range(len(acts_per_layer)):
    print(layer_idx)
    acts_this_layer = [torch.from_numpy(np.load(a)).float() for a in acts_per_layer[layer_idx]]
    results = []
    for sample_idx in range(len(acts_this_layer)):
        results.append(linear_cka_2d(acts_this_layer[sample_idx], noise[sample_idx]))
    print('noise', np.mean(results))
    results = []
    for sample_idx in range(len(acts_this_layer)):
        results.append(linear_cka_2d(acts_this_layer[sample_idx], target[sample_idx]))
    print('target', np.mean(results))
