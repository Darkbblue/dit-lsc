import os
import tqdm
import glob
import json
import torch
import argparse
import numpy as np
from PIL import Image
from mine_runner import compute_layerwise_mi

# args
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=4000)
parser.add_argument('--feat_root', type=str)
parser.add_argument('--img_root', type=str)
parser.add_argument('--layers', type=str)
parser.add_argument('--critic_hidden', type=int, default=512)
parser.add_argument('--proj_dim', type=int, default=256)
parser.add_argument('--train_steps', type=int, default=600)

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

# for layer in acts_per_layer:
#     print(layer.shape)
# print()
# print(noise.shape)
# print(target.shape)
# exit()


# compute
print('start computing')
mi_curves = compute_layerwise_mi(
    acts_per_layer=acts_per_layer,
    noise=noise,
    target=target,
    proj_dim=args.proj_dim,
    train_steps=args.train_steps,   # bump up (e.g., 1000+) for final figures
    critic_hidden=args.critic_hidden,
    device="cuda",
)

# print(mi_curves)  # dict with lists per layer
for i in range(len(layers)):
    print(layers[i])
    print('MI_AT', mi_curves['MI_AT'][i])
    print('MI_AN', mi_curves['MI_AN'][i])
    print('MI_ANT', mi_curves['MI_ANT'][i])
    print('MI_AN_given_T', mi_curves['MI_AN_given_T'][i])
