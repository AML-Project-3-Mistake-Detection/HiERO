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

Output:
  results.json  — step intervals per video, mirroring the annotation structure:
  {
    "1_7": {
      "recording_id": "1_7",
      "steps": [
        {"step_id": 2, "start_time": 5.0,  "end_time": 38.0},
        {"step_id": 0, "start_time": 55.0, "end_time": 120.0},
        ...
      ]
    }
  }

  embeddings.npz  — step-level EgoVLP embeddings (averaged over each step's
  time range). Contains one array per recording_id, shape [num_steps, 256].
  Load with: data = np.load('embeddings.npz'); emb = data['1_7']  # [S, 256]

Step-level embeddings: for each detected step (start_time, end_time), the
original EgoVLP features (1 feature/second) that fall within that range are
averaged into a single 256-d vector. This is the step-level representation
used for downstream recipe understanding tasks.

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
from scipy.stats import truncnorm
from torch_geometric.data import Data

from egoprocel.utils import clusterize

# Ground-truth statistics for steps-per-video (from dataset annotations)
_GT_STEPS_MEAN   = 14.84
_GT_STEPS_STD    = 4.38
_GT_STEPS_MIN    = 7
_GT_STEPS_MAX    = 26


def sample_num_steps(rng: np.random.Generator) -> int:
    """Sample a plausible number of steps from the ground-truth distribution.

    Uses a truncated normal (mean=14.1, std=4.36) clipped to [5, 25].
    """
    a = (_GT_STEPS_MIN - _GT_STEPS_MEAN) / _GT_STEPS_STD
    b = (_GT_STEPS_MAX - _GT_STEPS_MEAN) / _GT_STEPS_STD
    value = truncnorm.rvs(a, b, loc=_GT_STEPS_MEAN, scale=_GT_STEPS_STD,
                          random_state=rng)
    return int(round(value))


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
        pos     = torch.arange(N, device=device, dtype=torch.float) * (stride / fps)
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


def compute_step_embeddings(raw_features: torch.Tensor, steps: list, fps: float) -> np.ndarray:
    """Average the raw EgoVLP features within each step's time boundaries.

    Parameters
    ----------
    raw_features : Tensor[N, feature_dim]  — original features (1 per second at fps=1)
    steps        : list of {start_time, end_time, ...} dicts
    fps          : features per second of raw_features

    Returns
    -------
    ndarray[S, feature_dim]  — one averaged embedding per step
    """
    feat = raw_features.cpu().numpy()  # [N, D]
    embeddings = []
    for step in steps:
        i_start = int(step["start_time"] * fps)
        i_end   = max(i_start + 1, int(step["end_time"] * fps))
        i_end   = min(i_end, len(feat))
        embeddings.append(feat[i_start:i_end].mean(axis=0))
    return np.stack(embeddings) if embeddings else np.empty((0, feat.shape[1]))


def _smooth_labels(labels: np.ndarray, window: int) -> np.ndarray:
    """Replace each position with the mode label in a sliding window."""
    if window <= 1 or len(labels) <= window:
        return labels.copy()
    smoothed = np.empty_like(labels)
    half = window // 2
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        vals, counts = np.unique(labels[lo:hi], return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    return smoothed


def labels_to_steps(labels: np.ndarray, seg_duration: float, smooth_window: int = 5) -> list:
    """
    Convert per-segment cluster labels into one interval per step.

    The label sequence is first smoothed with a mode filter to consolidate
    scattered noise assignments into coherent regions.  Each cluster's interval
    is then its **longest contiguous run** in the smoothed sequence.  With
    temporally-constrained agglomerative clustering each cluster is already a
    single contiguous block; the run heuristic here handles any small fragments
    introduced by the mode-smoothing pass without bridging gaps across time.

    Parameters
    ----------
    labels : ndarray[M]  — integer cluster label; -1 means background/no-step
    seg_duration : float — duration in seconds of each decoded segment
    smooth_window : int  — mode-filter window size (in segments)

    Returns
    -------
    List of dicts: [{"step_id": int, "start_time": float, "end_time": float}, ...]
    one entry per step, sorted by start_time, guaranteed non-overlapping.
    """
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    if not unique_clusters:
        return []

    smoothed = _smooth_labels(labels, window=smooth_window)

    steps = []
    for c in unique_clusters:
        idxs = np.where(smoothed == c)[0]
        if len(idxs) == 0:
            # Cluster was completely smoothed away; fall back to raw labels.
            idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue

        # Find the longest contiguous run of this cluster label.
        # With temporally-constrained clustering each cluster is already a single
        # contiguous block, but the mode-smoothing pass may fragment it slightly;
        # taking the longest run avoids capturing scattered noise segments.
        best_start = best_end = cur_start = idxs[0]
        for prev, cur in zip(idxs, idxs[1:]):
            if cur == prev + 1:
                best_end = cur
            else:
                if (best_end - cur_start) < (cur - cur_start):
                    best_start, best_end = cur_start, prev
                cur_start = cur
                best_end = max(best_end, prev)
        # Handle the last run
        if (cur_start <= idxs[-1]) and (idxs[-1] - cur_start) >= (best_end - best_start):
            best_start, best_end = cur_start, idxs[-1]

        steps.append({
            "step_id":    int(c),
            "start_time": round(int(best_start) * seg_duration, 3),
            "end_time":   round((int(best_end) + 1) * seg_duration, 3),
        })

    steps.sort(key=lambda x: x["start_time"])

    # Resolve any residual overlaps after smoothing.
    for i in range(len(steps) - 1):
        if steps[i]["end_time"] > steps[i + 1]["start_time"]:
            steps[i]["end_time"] = steps[i + 1]["start_time"]

    return [s for s in steps if s["end_time"] > s["start_time"]]


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
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Fix the number of steps for all videos. If not set, samples per-video "
                             "from a truncated normal (mean=14.1, std=4.36, min=5, max=25) "
                             "matching ground-truth statistics.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for per-video step count sampling")
    parser.add_argument("--steps_config", default=None,
                        help="JSON file mapping video filenames to their specific step count")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Mode-filter window (in decoded segments) for temporal smoothing before "
                             "extracting step intervals. Larger = smoother boundaries. Default 5 ≈ 20s.")
    parser.add_argument("--no_background", action="store_true",
                        help="Disable background cluster detection (every segment is assigned to a step)")
    parser.add_argument("--temp", type=float, default=0.05,
                        help="Temperature for spectral clustering affinity kernel")
    parser.add_argument("--use_proj_head", action="store_true",
                        help="Use the language-aligned projection head")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results.json",
                        help="Output JSON file with step intervals")
    parser.add_argument("--embeddings_output", default="embeddings.npz",
                        help="Output .npz file with step-level EgoVLP embeddings (shape [S, 256] per video)")
    args = parser.parse_args()

    # Load per-video step counts if provided
    per_video_steps = {}
    if args.steps_config is not None:
        with open(args.steps_config) as f:
            per_video_steps = json.load(f)
        print(f"Loaded per-video step counts for {len(per_video_steps)} videos from '{args.steps_config}'")

    rng = np.random.default_rng(args.seed)
    if args.num_steps is None:
        print(f"No --num_steps set: will sample per-video from "
              f"truncated normal (mean={_GT_STEPS_MEAN}, std={_GT_STEPS_STD}, "
              f"range=[{_GT_STEPS_MIN}, {_GT_STEPS_MAX}]) with seed={args.seed}")
    else:
        print(f"Using fixed num_steps={args.num_steps} for all videos")

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
    all_embeddings = {}  # recording_id → ndarray [S, 256]
    for npz_path in npz_files:
        video_name = os.path.basename(npz_path)
        recording_id = os.path.splitext(video_name)[0].replace("_360p_224.mp4_1s_1s", "")
        if recording_id in results:
            print(f"  WARNING: {video_name}: duplicate recording_id '{recording_id}', skipping.")
            continue
        features = load_npz_features(npz_path)          # [N, 256]

        # Per-video step count: explicit config > fixed CLI arg > sampled from GT distribution
        if video_name in per_video_steps:
            num_steps = int(per_video_steps[video_name])
        elif args.num_steps is not None:
            num_steps = args.num_steps
        else:
            num_steps = sample_num_steps(rng)

        # Run HiERO temporal backbone
        segment_features = features_extractor(features)  # [M, hidden_size]
        M = segment_features.shape[0]

        if M == 0:
            print(f"  WARNING: {video_name}: no decoded segments produced, skipping.")
            results[recording_id] = {"recording_id": recording_id, "steps": []}
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

        # One interval per step (longest contiguous run after smoothing), sorted by start_time
        steps = labels_to_steps(step_labels, seg_duration, smooth_window=args.smooth_window)

        # Step-level embeddings: average raw EgoVLP features within each step's boundaries
        step_embeddings = compute_step_embeddings(features, steps, fps=args.fps)

        results[recording_id] = {
            "recording_id": recording_id,
            "steps": steps,
        }
        all_embeddings[recording_id] = step_embeddings

        n_bg = int((step_labels == -1).sum()) if use_background else 0
        print(f"  {video_name}: {features.shape[0]} input segs → "
              f"{M} decoded segs ({num_steps} steps requested) → {len(steps)} steps found "
              f"({n_bg} background segs = {n_bg * seg_duration:.0f}s of gaps)")
        for s in steps:
            print(f"    step_id {s['step_id']:2d}  [{s['start_time']:7.1f}s – {s['end_time']:7.1f}s]")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Step intervals saved to '{args.output}'")

    np.savez(args.embeddings_output, **all_embeddings)
    print(f"Step embeddings saved to '{args.embeddings_output}' "
          f"({len(all_embeddings)} videos, use np.load('{args.embeddings_output}') to load)")


if __name__ == "__main__":
    main()
