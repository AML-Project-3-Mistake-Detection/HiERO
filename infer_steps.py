"""
Identify procedure steps in videos using the pretrained HiERO model.

Usage:
    python infer_steps.py [--feature_dir video_features] [--ckpt pretrained/hiero_egovlp.pth]
                          [--num_steps 7] [--steps_config steps.json]
                          [--depth 2] [--temp 0.5] [--use_proj_head]
                          [--output results.json]

Each .npz file in `feature_dir` is expected to contain an 'arr_0' array of shape
(N, 256) with EgoVLP features (one feature vector per second of video).

--num_steps sets the default number of steps for all videos.
--steps_config (optional) is a JSON file mapping video filenames to their specific
  step count, e.g.: {"1_10_360p_224.mp4_1s_1s.npz": 5, "1_14_360p_224.mp4_1s_1s.npz": 9}
  Per-video values override --num_steps.

Output format (results.json) mirrors the ground-truth annotation structure:
{
  "1_7": {
    "recording_id": "1_7",
    "steps": [
      {"step_id": 2, "start_time": 5.0,   "end_time": 38.0},
      {"step_id": 0, "start_time": 55.0,  "end_time": 120.0},
      {"step_id": 3, "start_time": 145.0, "end_time": 200.0},
      ...
    ]
  }
}
Each entry is one step, ordered by start_time. Each step appears exactly once.
The interval [start_time, end_time] spans all segments of that cluster.
Gaps between entries are background. step_id values are arbitrary cluster IDs
assigned by the model (they won't match ground-truth step numbers, but the
temporal segmentation is what matters).

Background detection: by default, num_steps+1 clusters are found and the
least coherent one (lowest mean intra-cluster cosine similarity) is treated
as background and excluded from the output. Disable with --no_background.
"""

import argparse
import json
import os
import glob

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from egoprocel.utils import clusterize


def build_hiero_model(ckpt: str, fps: float, depth: int, use_proj_head: bool,
                      input_size: int, stride: int, device: str):
    """Load a HiERO model from a checkpoint and return a features_extractor callable."""
    weights = torch.load(ckpt, weights_only=False, map_location=device)

    model = hydra.utils.instantiate(
        weights["config"]["model"],
        clustering_at_inference="active",
        input_size=input_size,
        _recursive_=False,
    ).to(device)

    task = hydra.utils.instantiate(
        weights["config"]["task"],
        _recursive_=False,
    ).to(device)

    model.load_state_dict(weights["model"], strict=True)
    task.load_state_dict(weights["task"], strict=True)
    model.eval()
    task.eval()

    # Temporal length (in seconds) of each decoded node at the chosen decoder depth
    node_length = (stride / fps) * (2 ** depth)

    @torch.no_grad()
    def features_extractor(features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : Tensor[N, input_size]
        Returns
        -------
        Tensor[M, hidden_size]  — one feature per decoded node at `depth`
        """
        N = features.shape[0]
        pos     = torch.arange(N, device=device, dtype=torch.float) * node_length
        indices = torch.arange(N, device=device)
        batch   = torch.zeros(N, dtype=torch.long, device=device)
        mask    = torch.ones(N, dtype=torch.bool, device=device)

        data = Data(
            x=features.unsqueeze(1).to(device),
            pos=pos,
            indices=indices,
            batch=batch,
            mask=mask,
        )

        graphs = model(data)

        if use_proj_head:
            out = task(graphs, data)
        else:
            out = graphs.x

        return out[graphs.depth == depth]

    return features_extractor, node_length


def load_npz_features(path: str) -> torch.Tensor:
    """Load EgoVLP features from an .npz file → float32 Tensor [N, 256]."""
    d = np.load(path)
    key = list(d.keys())[0]          # typically 'arr_0'
    return torch.from_numpy(d[key].astype(np.float32))


def identify_background_cluster(features: torch.Tensor, labels: np.ndarray, n: int) -> int:
    """
    Return the cluster index most likely to be background.

    The background cluster is the one with the lowest mean intra-cluster
    cosine similarity — its members look least like each other, which is
    characteristic of heterogeneous transition/gap segments.
    """
    feats = F.normalize(features.cpu().float(), p=2, dim=-1)
    mean_sim = []
    for c in range(n):
        mask = labels == c
        if mask.sum() <= 1:
            mean_sim.append(-1.0)   # single-member: maximally incoherent
            continue
        cluster_feats = feats[mask]
        sim = (cluster_feats @ cluster_feats.T).mean().item()
        mean_sim.append(sim)
    return int(np.argmin(mean_sim))


def labels_to_steps(labels: np.ndarray, seg_duration: float) -> list:
    """
    Convert per-segment cluster labels into one interval per step.

    Each cluster (excluding -1 background) produces one entry. Entries are
    sorted by the first time the cluster appears in the video. The interval
    [start_time, end_time] is the bounding box of all segments in the cluster.

    Parameters
    ----------
    labels : ndarray[M]  — integer cluster label; -1 means background/no-step
    seg_duration : float — duration in seconds of each decoded segment

    Returns
    -------
    List of dicts: [{"step_id": int, "start_time": float, "end_time": float}, ...]
    one entry per step, sorted by start_time.
    """
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    if not unique_clusters:
        return []

    cluster_info = []
    for c in unique_clusters:
        idxs = np.where(labels == c)[0]
        start = round(float(idxs.min()) * seg_duration, 3)
        end   = round(float(idxs.max() + 1) * seg_duration, 3)
        cluster_info.append((start, end, int(c)))

    # Sort by start_time (first appearance in the video)
    cluster_info.sort(key=lambda x: x[0])

    return [
        {"step_id": step_id, "start_time": start, "end_time": end}
        for start, end, step_id in cluster_info
    ]


def main():
    parser = argparse.ArgumentParser(description="HiERO step identification")
    parser.add_argument("--feature_dir", default="video_features",
                        help="Directory containing .npz feature files")
    parser.add_argument("--ckpt", default="pretrained/hiero_egovlp.pth",
                        help="Path to HiERO checkpoint")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Feature dimensionality (256 for EgoVLP, 1536 for Omnivore)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Features per second (1.0 for 1s-stride features)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Feature stride (in frames, consistent with fps)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Decoder depth level to extract features from (0=finest)")
    parser.add_argument("--num_steps", type=int, default=14,
                        help="Default number of procedure steps per video (ground-truth median=14, mean=14.1)")
    parser.add_argument("--steps_config", default=None,
                        help="JSON file mapping video filenames to their specific step count")
    parser.add_argument("--no_background", action="store_true",
                        help="Disable background cluster detection (every segment is assigned to a step)")
    parser.add_argument("--temp", type=float, default=0.5,
                        help="Temperature for spectral clustering affinity kernel")
    parser.add_argument("--use_proj_head", action="store_true",
                        help="Use the language-aligned projection head")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results.json",
                        help="Output JSON file with step intervals")
    args = parser.parse_args()

    # Load per-video step counts if provided
    per_video_steps = {}
    if args.steps_config is not None:
        with open(args.steps_config) as f:
            per_video_steps = json.load(f)
        print(f"Loaded per-video step counts for {len(per_video_steps)} videos from '{args.steps_config}'")

    print(f"Device : {args.device}")
    print(f"Loading model from {args.ckpt}...")
    features_extractor, seg_duration = build_hiero_model(
        ckpt=args.ckpt,
        fps=args.fps,
        depth=args.depth,
        use_proj_head=args.use_proj_head,
        input_size=args.input_size,
        stride=args.stride,
        device=args.device,
    )
    print(f"Each decoded segment covers {seg_duration:.2f}s of source video\n")

    npz_files = sorted(glob.glob(os.path.join(args.feature_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in '{args.feature_dir}'")
        return

    print(f"Found {len(npz_files)} feature files. Running HiERO inference...\n")

    results = {}
    for npz_path in npz_files:
        video_name = os.path.basename(npz_path)
        features = load_npz_features(npz_path)          # [N, 256]

        # Per-video step count overrides the global default
        num_steps = per_video_steps.get(video_name, args.num_steps)

        # Run HiERO temporal backbone
        segment_features = features_extractor(features)  # [M, hidden_size]
        M = segment_features.shape[0]

        if M == 0:
            print(f"  WARNING: {video_name}: no decoded segments produced, skipping.")
            results[video_name] = []
            continue

        # When using background detection we cluster into num_steps+1 groups;
        # the extra cluster absorbs the heterogeneous gap/transition segments.
        use_background = not args.no_background
        n_clusters = num_steps + 1 if use_background else num_steps

        if M < n_clusters:
            print(f"  WARNING: {video_name}: only {M} decoded segments but {n_clusters} clusters "
                  f"requested. Clamping to {M}.")
            n_clusters = M
            if use_background and n_clusters <= 1:
                use_background = False

        # Spectral clustering → step label per decoded segment
        step_labels = clusterize(segment_features, n=n_clusters, temp=args.temp)

        if use_background:
            bg = identify_background_cluster(segment_features, step_labels, n_clusters)
            # Mark background segments as -1; keep all other raw cluster IDs
            step_labels = np.where(step_labels == bg, -1, step_labels)

        # One interval per step, sorted by start_time
        steps = labels_to_steps(step_labels, seg_duration)
        recording_id = os.path.splitext(video_name)[0].replace("_360p_224.mp4_1s_1s", "")
        results[recording_id] = {
            "recording_id": recording_id,
            "steps": steps,
        }

        n_bg = int((step_labels == -1).sum()) if use_background else 0
        print(f"  {video_name}: {features.shape[0]} input segs → "
              f"{M} decoded segs → {len(steps)} steps "
              f"({n_bg} background segs = {n_bg * seg_duration:.0f}s of gaps)")
        for s in steps:
            print(f"    step_id {s['step_id']:2d}  [{s['start_time']:7.1f}s – {s['end_time']:7.1f}s]")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to '{args.output}'")


if __name__ == "__main__":
    main()
