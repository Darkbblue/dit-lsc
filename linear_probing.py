import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, r2_score

# The diffusion_feature library lives under ./feature, so the feature directory
# must be added to sys.path so that `from components.models import ...` inside
# diffusion_feature.py resolves correctly
_FEATURE_DIR = (Path(__file__).resolve().parent / "feature")
if str(_FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(_FEATURE_DIR))

import diffusion_feature


class DiTFeatureDataset(Dataset):
    """
    Both inputs and targets are extracted from the DiT model via the
    diffusion_feature library.

    Multiple input layers (layers) and multiple target layers (target_layers)
    can be configured at once; set_pair(input_layer, target_layer) switches the
    currently active (x, y) pair. A single forward pass collects the features
    of all configured layers and caches them per layer, so different pairs can
    reuse the same feature cache and avoid redundant extraction.

    Data layout:
        data_dir/
        ├── train/
        │   ├── 0000.png
        │   └── ...
        └── val/
            ├── 0000.png
            └── ...

    In __getitem__:
        1. Fetch the input-layer feature x and the target-layer feature y for
           the current pair, both of shape (c, h, w);
        2. Align y's spatial size to x, then reshape both to (h*w, c),
           treating h*w as the batch size.

    If cache_dir is specified, features are cached as .pt files at
    (image, layer, t) granularity; otherwise this falls back to an in-process
    memory cache (avoiding redundant forward passes per pair/epoch).
    """

    def __init__(
        self,
        data_dir,
        split,
        feature_extractor,
        layers,
        t,
        target_layers=None,
        target_t=None,
        cache_dir=None,
    ):
        self.image_dir = Path(data_dir) / split

        self.image_paths = sorted(self.image_dir.glob("*.png")) + \
                           sorted(self.image_dir.glob("*.jpg")) + \
                           sorted(self.image_dir.glob("*.JPEG"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found under {self.image_dir}")

        self.feature_extractor = feature_extractor
        self.layers = list(layers)
        self.t = t
        self.target_layers = list(target_layers) if target_layers else list(self.layers)
        self.target_t = target_t if target_t is not None else t

        # Currently active pair, switchable via set_pair
        self.input_layer = self.layers[0]
        self.target_layer = self.target_layers[0]

        self.cache_dir = Path(cache_dir) / split if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache = {}

        self.target_size = None

    def set_pair(self, input_layer, target_layer):
        """Switch the current (input layer, target layer) pair."""
        self.input_layer = input_layer
        self.target_layer = target_layer

    def __len__(self):
        return len(self.image_paths)

    def _cache_path(self, image_path, layer, t):
        safe_layer = layer.replace("/", "_")
        return self.cache_dir / f"{image_path.stem}_{safe_layer}_t{t}.pt"

    def _load_cached(self, image_path, layer, t):
        if self.cache_dir:
            path = self._cache_path(image_path, layer, t)
            if not path.exists():
                return None
            try:
                return torch.load(path, map_location="cpu", weights_only=False).float()
            except TypeError:
                return torch.load(path, map_location="cpu", weights_only=False).float()
        return self._mem_cache.get((str(image_path), layer, t))

    def _save_cached(self, image_path, layer, t, feat):
        if self.cache_dir:
            torch.save(feat, self._cache_path(image_path, layer, t))
        else:
            self._mem_cache[(str(image_path), layer, t)] = feat

    def _extract_all_at_t(self, image_path, t):
        """Run one forward pass for a single image at timestep t and cache the features of all configured layers."""
        image = Image.open(image_path).convert("RGB")

        # No gradients are needed during feature extraction; disable them explicitly
        # to avoid any gradient leftovers or wasted GPU memory
        with torch.no_grad():
            stored_feats = self.feature_extractor.extract(
                prompts='prompts',
                batch_size=1,
                image=[image],
                t=t,
                image_type="image",
            )

            wanted = set(self.layers) | set(self.target_layers)
            missing = wanted - set(stored_feats.keys())
            if missing:
                available = list(stored_feats.keys())
                raise KeyError(f"Requested layers {sorted(missing)} are not in the extraction result; available layers: {available}")

            for layer in wanted:
                feat = stored_feats[layer][0].float().cpu().detach()  # (c, h, w)
                self._save_cached(image_path, layer, t, feat)

    def _get_feature(self, image_path, layer, t):
        feat = self._load_cached(image_path, layer, t)
        if feat is None:
            self._extract_all_at_t(image_path, t)
            feat = self._load_cached(image_path, layer, t)
        return feat

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        x = self._get_feature(image_path, self.input_layer, self.t)
        if self.target_layer == self.input_layer and self.target_t == self.t:
            y = x.clone()
        else:
            y = self._get_feature(image_path, self.target_layer, self.target_t)

        # Align the spatial size of the target feature to the input feature
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y.unsqueeze(0), size=x.shape[-2:], mode="bilinear", align_corners=False)[0]

        self.target_size = (x.shape[1], x.shape[2])

        c, h, w = x.shape
        x = x.reshape(c, h * w).permute(1, 0)  # (h*w, c_x)

        cy = y.shape[0]
        y = y.reshape(cy, h * w).permute(1, 0)  # (h*w, c_y)

        return x, y


class LinearProbe(nn.Module):
    """Performs an independent linear transformation per pixel/patch: y = xW + b"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        # x: (b, h*w, c)
        return self.linear(x)


def collate_fn(batch):
    """Concatenate the spatial positions of all images within a batch: list[(h*w, c)] -> (sum(h*w), c)"""
    xs, ys = zip(*batch)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_epoch(model, loader, criterion, optimizer, device, task="regression"):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        pred = model(x)  # (sum(h*w), out_c)

        if task == "classification":
            if y.dim() == 2 and y.shape[1] == pred.shape[1]:
                y_idx = y.argmax(dim=-1).long()
            else:
                y_idx = y.long().squeeze(-1)
            y_idx = y_idx.to(device)
            loss = criterion(pred, y_idx)
        else:
            y = y.to(device)
            loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, task="regression"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        pred = model(x)

        if task == "classification":
            if y.dim() == 2 and y.shape[1] == pred.shape[1]:
                y_idx = y.argmax(dim=-1).long()
            else:
                y_idx = y.long().squeeze(-1)
            y_idx = y_idx.to(device)
            loss = criterion(pred, y_idx)
            all_preds.append(pred.argmax(dim=-1).cpu().numpy())
            all_targets.append(y_idx.cpu().numpy())
        else:
            y = y.to(device)
            loss = criterion(pred, y)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    result = {"loss": total_loss / total_samples}

    if task == "classification":
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        result["accuracy"] = accuracy_score(targets, preds)
        result["f1_macro"] = f1_score(targets, preds, average="macro", zero_division=0)
    else:
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        result["r2"] = r2_score(targets, preds, multioutput="uniform_average")

    return result


def build_layer_config(layers):
    """Convert multiple layer strings or JSON paths into the config dict expected
    by FeatureExtractor, and also return the normalized layer-name list (input order preserved)."""
    config = {}
    names = []
    for layer in layers:
        layer_path = Path(layer)
        if layer_path.exists() and layer_path.suffix == ".json":
            with open(layer_path, "r") as f:
                sub = json.load(f)
            config.update(sub)
            names.extend(k for k in sub.keys() if k not in names)
            continue
        # Inside the DiT model the gather uses module_id 'vit', so the final key looks like 'vit-block10-out'
        if not layer.startswith("vit-") and not layer.startswith("pipe-"):
            layer = "vit-" + layer
        config[layer] = True
        if layer not in names:
            names.append(layer)
    return config, names


def run_pair(args, train_ds, val_ds, input_layer, target_layer):
    """Train a linear probe for one (input layer, target layer) pair and return the best val metrics."""
    train_ds.set_pair(input_layer, target_layer)
    val_ds.set_pair(input_layer, target_layer)

    # Use the first sample to determine input/output dimensions
    x0, y0 = train_ds[0]
    in_channels, out_channels = x0.shape[-1], y0.shape[-1]
    print(f"\n===== Pair: {input_layer} -> {target_layer} | in_c={in_channels}, out_c={out_channels} =====")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    model = LinearProbe(in_channels, out_channels).to(args.device)

    if args.task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device, task=args.task)
        val_metrics = evaluate(model, val_loader, criterion, args.device, task=args.task)
        scheduler.step()

        info = f"Epoch {epoch:03d} | train_loss: {train_loss:.6f} | val_loss: {val_metrics['loss']:.6f}"
        if args.task == "classification":
            info += f" | acc: {val_metrics['accuracy']:.4f} | f1: {val_metrics['f1_macro']:.4f}"
        else:
            info += f" | val_r2: {val_metrics['r2']:.4f}"
        print(info)

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = val_metrics
            if args.save_dir:
                tag = f"{input_layer}__to__{target_layer}".replace("/", "_")
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_linear_probe_{tag}.pt"))

    return best_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Linear probing on DiT features: extract multiple input/target (c, h, w) representations from DiT and fit per-pixel linear probes for every (input_layer, target_layer) pair."
    )
    # Data and IO
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory, must contain {train,val}/ subdirectories with images placed directly inside")
    parser.add_argument("--cache_dir", type=str, default=None, help="Feature cache directory, defaults to no disk cache (an in-process memory cache is used instead; disk caching is strongly recommended)")
    # Model and feature extraction (input)
    parser.add_argument("--version", type=str, default="dit w/ lsc", help="DiT version")
    parser.add_argument("--layer", type=str, nargs="+", default=["vit-block10-out"], help="Input feature layer names, multiple allowed, e.g. vit-block10-out block14-out")
    parser.add_argument("--t", type=int, default=50, help="Diffusion timestep for input features")
    # Model and feature extraction (target)
    parser.add_argument("--target_layer", type=str, nargs="+", default=None, help="Target feature layer names, multiple allowed, defaults to --layer")
    parser.add_argument("--target_t", type=int, default=None, help="Diffusion timestep for target features, defaults to --t")
    # Other model settings
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--feature_resize", type=int, default=1, help="Feature downsampling factor")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Linear probing hyperparameters
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8, help="Image-level batch size (h*w spatial positions are further concatenated within each worker)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--result_path", type=str, default=None, help="Path to save the R2 result matrix JSON, defaults to save_dir/r2_matrix.json")
    parser.add_argument("--out_channels", type=int, default=1, help="Number of output channels")
    args = parser.parse_args()

    # Initialize the DiT feature extractor: register all input and target layers at once
    input_config, input_layers = build_layer_config(args.layer)
    if args.target_layer:
        target_config, target_layers = build_layer_config(args.target_layer)
    else:
        target_config, target_layers = {}, list(input_layers)
    layer_config = {**input_config, **target_config}

    print(f"Loading DiT feature extractor: version={args.version}")
    print(f"Input layers: {input_layers}")
    print(f"Target layers: {target_layers}")
    df = diffusion_feature.FeatureExtractor(
        layer=layer_config,
        version=args.version,
        device=args.device,
        dtype=args.dtype,
        img_size=args.img_size,
        feature_resize=args.feature_resize,
    )

    # Build the datasets: online extraction + per-layer caching (disk or memory), cache shared across pairs
    # Note: FeatureExtractor holds a pipe on the GPU and cannot be serialized into
    # DataLoader worker processes, hence num_workers=0 here.
    train_ds = DiTFeatureDataset(
        args.data_dir,
        "train",
        feature_extractor=df,
        layers=input_layers,
        t=args.t,
        target_layers=target_layers,
        target_t=args.target_t,
        cache_dir=args.cache_dir,
    )
    val_ds = DiTFeatureDataset(
        args.data_dir,
        "test",
        feature_extractor=df,
        layers=input_layers,
        t=args.t,
        target_layers=target_layers,
        target_t=args.target_t,
        cache_dir=args.cache_dir,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Train all pairwise combinations and collect metrics
    results = {}
    for input_layer in input_layers:
        for target_layer in target_layers:
            best_metrics = run_pair(args, train_ds, val_ds, input_layer, target_layer)
            results[f"{input_layer} -> {target_layer}"] = best_metrics

    # Summarize results
    metric_key = "r2" if args.task == "regression" else "accuracy"
    print("\n========== Pairwise Result Summary ==========")
    col_w = max(len(l) for l in target_layers) + 2
    header = " " * (max(len(l) for l in input_layers) + 2) + "".join(l.ljust(col_w) for l in target_layers)
    print(f"Metric: {metric_key}")
    print(header)
    for input_layer in input_layers:
        row = input_layer.ljust(max(len(l) for l in input_layers) + 2)
        for target_layer in target_layers:
            m = results[f"{input_layer} -> {target_layer}"]
            row += f"{m[metric_key]:.4f}".ljust(col_w)
        print(row)

    result_path = args.result_path or os.path.join(args.save_dir, "r2_matrix.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "task": args.task,
                "metric": metric_key,
                "t": args.t,
                "target_t": args.target_t if args.target_t is not None else args.t,
                "input_layers": input_layers,
                "target_layers": target_layers,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
