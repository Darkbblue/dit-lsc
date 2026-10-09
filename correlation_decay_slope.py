import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse the diffusion_feature library from the generic-diffusion-feature project
_DIFFUSION_FEATURE_DIR = Path(__file__).resolve().parents[2] / "generic-diffusion-feature" / "feature"
if str(_DIFFUSION_FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_FEATURE_DIR))

import diffusion_feature


def build_layer_config(layers):
    """Convert multiple layer strings into the config dict required by FeatureExtractor."""
    config = {}
    for layer in layers:
        layer_path = Path(layer)
        if layer_path.exists() and layer_path.suffix == ".json":
            with open(layer_path, "r") as f:
                config.update(json.load(f))
            continue
        if not layer.startswith("vit-") and not layer.startswith("pipe-"):
            layer = "vit-" + layer
        config[layer] = True
    return config


def extract_features(image_dir, feature_extractor, layers, t):
    """
    Extract DiT features for multiple layers from an image directory in one pass.

    Returns: list of {"image": Path, "feats": {layer_name: tensor (c, h, w)}}
    """
    image_paths = sorted(Path(image_dir).glob("*.png")) + \
                  sorted(Path(image_dir).glob("*.jpg")) + \
                  sorted(Path(image_dir).glob("*.JPEG"))
    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found under {image_dir}")

    results = []
    # prompts = feature_extractor.encode_prompt(None)
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            stored_feats = feature_extractor.extract(
                prompts='prompts',
                batch_size=1,
                image=[image],
                t=t,
                image_type="image",
            )

        sample = {"image": image_path, "feats": {}}
        for layer in layers:
            if layer not in stored_feats:
                available = list(stored_feats.keys())
                raise KeyError(f"Requested layer {layer} not in extraction results, available layers: {available}")
            sample["feats"][layer] = stored_feats[layer][0].float().cpu().detach()
        results.append(sample)

    return results


def compute_correlation_decay(feat, metric="cosine", min_pairs=100, min_distance_bins=3):
    """
    Compute the spatial correlation decay of a single feature map (c, h, w).

    Steps:
        1. Flatten the feature at each spatial position into a vector, giving (h*w, c);
        2. Compute the similarity between all distinct position pairs (cosine similarity or Pearson correlation);
        3. Compute the corresponding spatial Euclidean distances;
        4. Group and average by integer distance to obtain the full per-distance correlation curve;
        5. Fit a line in log-log space and return a single slope value.

    Returns:
        slope: slope of the linear fit of log(correlation) ~ log(distance)
        intercept: intercept of the fit
        distances: array of integer distances (in ascending order)
        correlations: array of the average correlation at each distance
    """
    c, h, w = feat.shape
    feat = feat.reshape(c, h * w).permute(1, 0)  # (n, c), n = h*w
    n = h * w

    if n * (n - 1) // 2 < min_pairs:
        return None, None, [], []

    # Spatial coordinates
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()  # (n, 2)

    # Compute pairwise spatial distances
    dists = torch.cdist(coords, coords).numpy()  # (n, n)

    # Compute pairwise feature similarities
    if metric == "cosine":
        feat_norm = F.normalize(feat, dim=-1)
        corrs = (feat_norm @ feat_norm.t()).numpy()
    elif metric == "pearson":
        feat_centered = feat - feat.mean(dim=0, keepdim=True)
        feat_std = feat_centered.std(dim=0, keepdim=True) + 1e-8
        feat_normed = feat_centered / feat_std
        corrs = (feat_normed @ feat_normed.t()).numpy() / c
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Take only the upper triangle (excluding the diagonal) to avoid duplicates
    triu_idx = np.triu_indices(n, k=1)
    dists_flat = dists[triu_idx]
    corrs_flat = corrs[triu_idx]

    # Filter out non-positive correlation values to avoid log errors
    valid_mask = corrs_flat > 0
    if valid_mask.sum() < min_pairs:
        return None, None, [], []

    dists_flat = dists_flat[valid_mask]
    corrs_flat = corrs_flat[valid_mask]

    # Group by integer distance and compute the average correlation at each distance
    dist_int = np.rint(dists_flat).astype(int)
    unique_dists = np.unique(dist_int)
    mean_corrs = []
    for d in unique_dists:
        mean_corrs.append(corrs_flat[dist_int == d].mean())
    distances = unique_dists.astype(float)
    correlations = np.array(mean_corrs)

    # Need enough distance bins to fit a reliable slope
    if len(distances) < min_distance_bins:
        return None, None, distances.tolist(), correlations.tolist()

    # Global log-log linear fit to obtain a single decay slope
    log_dists = np.log(distances)
    log_corrs = np.log(correlations)
    slope, intercept = np.polyfit(log_dists, log_corrs, 1)

    return slope, intercept, distances.tolist(), correlations.tolist()


def aggregate_correlation_decay(all_distances, all_correlations):
    """
    Aggregate the distance-correlation curves across multiple images: average the
    correlations at the same integer distances to obtain the overall curve for the
    whole evaluation set, and re-fit the slope.
    """
    # Collect all (distance, correlation) pairs
    pairs = []
    for dists, corrs in zip(all_distances, all_correlations):
        for d, c in zip(dists, corrs):
            pairs.append((d, c))

    dist_arr = np.array([p[0] for p in pairs])
    corr_arr = np.array([p[1] for p in pairs])

    # Group and average by integer distance
    unique_dists = np.unique(dist_arr)
    mean_corrs = []
    for d in unique_dists:
        mean_corrs.append(corr_arr[dist_arr == d].mean())
    distances = unique_dists.astype(float)
    correlations = np.array(mean_corrs)

    # Fit the overall slope
    valid_mask = correlations > 0
    if valid_mask.sum() < 2:
        raise RuntimeError("Not enough valid data points to fit the slope")

    slope, intercept = np.polyfit(
        np.log(distances[valid_mask]), np.log(correlations[valid_mask]), 1
    )

    return slope, intercept, distances.tolist(), correlations.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Compute correlation decay slope of DiT features for multiple layers."
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--version", type=str, default="dit w/ lsc", choices=["dit w/o lsc", "dit w/ lsc"])
    parser.add_argument("--layer", type=str, nargs="+", default=["vit-block10-out"], help="Layer names to extract, multiple allowed")
    parser.add_argument("--t", type=int, default=50, help="Diffusion timestep")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--feature_resize", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "pearson"])
    parser.add_argument("--output", type=str, default="./correlation_decay_result.json", help="Path to save the results")
    args = parser.parse_args()

    # Normalize layer names
    normalized_layers = []
    for layer in args.layer:
        if not layer.startswith("vit-") and not layer.startswith("pipe-"):
            layer = "vit-" + layer
        normalized_layers.append(layer)

    layer_config = build_layer_config(normalized_layers)
    print(f"Loading DiT feature extractor: version={args.version}, layers={normalized_layers}, t={args.t}")
    df = diffusion_feature.FeatureExtractor(
        layer=layer_config,
        version=args.version,
        device=args.device,
        dtype=args.dtype,
        img_size=args.img_size,
        feature_resize=args.feature_resize,
    )

    print(f"Extracting features from {args.image_dir}...")
    samples = extract_features(args.image_dir, df, normalized_layers, args.t)

    # Aggregate results per layer
    layer_results = {}
    for layer in normalized_layers:
        all_distances = []
        all_correlations = []

        for sample in samples:
            feat = sample["feats"][layer]
            slope, intercept, distances, correlations = compute_correlation_decay(
                feat, metric=args.metric
            )
            if slope is None:
                print(f"Warning: {sample['image'].name} has too few valid position pairs on {layer}, skipping")
                continue

            all_distances.append(distances)
            all_correlations.append(correlations)

        if len(all_distances) == 0:
            print(f"Warning: no images were computed successfully for layer {layer}, skipping")
            continue

        overall_slope, overall_intercept, overall_distances, overall_correlations = \
            aggregate_correlation_decay(all_distances, all_correlations)

        layer_results[layer] = {
            "slope": float(overall_slope),
            "intercept": float(overall_intercept),
            "distances": overall_distances,
            "correlations": overall_correlations,
            "num_images": len(all_distances),
        }

        print(f"{layer}: slope={overall_slope:.4f} (based on {len(all_distances)} images)")

    if len(layer_results) == 0:
        raise RuntimeError("Failed to compute the correlation decay slope for any layer")

    result = {
        "t": args.t,
        "metric": args.metric,
        "layers": layer_results,
    }

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
