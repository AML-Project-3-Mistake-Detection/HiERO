# HiERO Step Identification Guide: Understanding `infer_steps.py`

## Overview

`infer_steps.py` is an inference pipeline that identifies procedure steps in egocentric videos. It uses a pretrained **HiERO** (Hierarchical Event Recognizing and Ordering) model to cluster temporal video features and segment them into meaningful procedure steps. The script takes pre-extracted EgoVLP features (256-dimensional embeddings) and outputs:
- **Step intervals** with timing information (when each step starts/ends)
- **Step-level embeddings** (averaged features representing each step)

---

## Key Concepts

### 1. **Input Features**
- **Format**: `.npz` files containing EgoVLP features (256-dim vectors)
- **Granularity**: One feature vector per second (1Hz sampling rate)
- **Shape**: `[N, 256]` where N = number of seconds in the video

### 2. **HiERO Model**
A hierarchical temporal model that:
- Processes features through multiple decoder depths (0 = finest, increasing = coarser)
- Groups temporal segments into clusters at each depth level
- At inference, extracts features at a specific decoder depth (default: depth=2)

### 3. **Step Segmentation**
- Uses **spectral clustering** to group similar temporal segments
- Detects a "background" cluster for gaps/transitions
- Converts cluster labels into contiguous step intervals with start/end times

---

## Execution Workflow

### Step 1: Parse Command-Line Arguments
```python
parser.add_argument("--feature_dir", default="video_features")
parser.add_argument("--num_steps", type=int, default=None)
parser.add_argument("--steps_config", default=None)
```
The script accepts:
- **Video feature directory** to process
- **Number of steps** (fixed or per-video via JSON config)
- **Model checkpoint** path
- **Decoder depth** and other HiERO parameters
- **Output file names** for results and embeddings

### Step 2: Load Per-Video Step Counts (Optional)
```python
if args.steps_config is not None:
    with open(args.steps_config) as f:
        per_video_steps = json.load(f)
```
If a JSON file maps videos to specific step counts, it overrides the `--num_steps` CLI argument. This allows flexible per-video configuration.

### Step 3: Initialize the HiERO Model
The `build_hiero_model()` function:
1. **Loads checkpoint weights** containing model and task configuration
2. **Instantiates the model** and task module using Hydra
3. **Moves to device** (GPU/CPU)
4. **Sets to eval mode**

**Returns a `features_extractor` callable** that processes raw features through the hierarchical decoder.

### Step 4: For Each Video File, Execute the Following:

#### 4a. Load Raw EgoVLP Features
```python
features = load_npz_features(npz_path)  # [N, 256]
```
Extracts 256-dim EgoVLP vectors from the `.npz` file (one vector/second).

#### 4b. Determine Number of Steps
Priority order:
1. **Per-video config** (if provided in `--steps_config`)
2. **Fixed CLI value** (if `--num_steps` is set)
3. **Sampled from distribution** (if neither above, samples from truncated normal with mean=14.84, std=4.38, range=[7, 26])

#### 4c. Extract Hierarchical Features
```python
segment_features = features_extractor(features)  # [M, hidden_size]
```
The `features_extractor`:
- Wraps raw features in PyTorch Geometric `Data` objects with temporal positions
- Processes through the hierarchical decoder graph network
- Returns features at the target decoder depth (each represents a ~20s segment at depth=2)
- **Output**: M segments where M < N (downsampling via hierarchy)

#### 4d. Perform Spectral Clustering
```python
step_labels = clusterize(segment_features, n=n_clusters, temp=args.temp)
```
Uses spectral clustering to group M segments into **n_clusters** clusters (where n_clusters = num_steps + 1 if background detection is enabled).

#### 4e. Identify Background Cluster (Optional)
```python
bg = identify_background_cluster(segment_features, step_labels, n_clusters)
step_labels = np.where(step_labels == bg, -1, step_labels)
```
The `identify_background_cluster()` function:
1. Computes intra-cluster cosine similarity for each cluster
2. Selects cluster with **lowest similarity** (most heterogeneous = background)
3. Marks those segments as `-1` to exclude from final output

Background clusters typically contain transition/gap segments that don't fit coherent procedure steps.

#### 4f. Convert Cluster Labels to Step Intervals
```python
steps = labels_to_steps(step_labels, seg_duration, smooth_window=args.smooth_window)
```
The `labels_to_steps()` function:

1. **Smooth labels** using mode filtering (sliding window to reduce noise)
   ```
   Window size ≈ 20s (default smooth_window=5 segments × ~4s/segment)
   ```

2. **Extract contiguous runs** of each cluster ID
   - Creates one step interval per contiguous run
   - Records step_id, start_time, end_time

3. **Merge fragmented steps** if too many exist
   - Limits fragmentation to expected_steps ≈ 1.25 × num_clusters
   - Merges shortest steps with nearest neighbors

4. **Filter out background** (step_id == -1)

5. **Enforce gap constraints**
   - Maximum gap between steps: 30 seconds (MAX_GAP = 16.0)
   - Distributes excess gap time between adjacent steps

#### 4g. Compute Step-Level Embeddings
```python
step_embeddings = compute_step_embeddings(features, steps, fps=1.0)
```
For each step interval:
- Identifies which raw EgoVLP frames fall within [start_time, end_time]
- Averages those frames into a single 256-dim vector
- Result: [num_steps, 256] array

#### 4h. Save Results Per Video
Stores in results dict:
- `recording_id`: cleaned video name
- `steps`: list of step intervals with timing
- `embeddings_shape`: metadata

---

## Key Functions Explained

### `sample_num_steps(rng: np.random.Generator) → int`
Samples a plausible number of steps from a truncated normal distribution matching ground-truth statistics:
- Mean = 14.84, Std = 4.38
- Clipped to [7, 26] range
- Ensures sampled values are realistic for the dataset

### `build_hiero_model(...) → (features_extractor, seg_duration, task)`
Creates a callable function that:
- Takes raw features [N, 256]
- Returns hierarchical features [M, hidden_size] at specified depth
- Each output segment covers ~20s of video (at depth=2)

### `identify_background_cluster(...) → int`
Returns index of the cluster most likely to be background by:
- Computing L2-normalized features
- Calculating mean cosine similarity within each cluster
- Selecting cluster with lowest similarity (least coherent)

### `labels_to_steps(...) → list[dict]`
Converts cluster labels into temporal intervals by:
1. Mode-filtering to smooth noise
2. Extracting contiguous label runs
3. Consolidating fragmentation
4. Enforcing maximum gap constraints
5. Filtering background (-1 labels)

### `compute_step_embeddings(...) → np.ndarray`
Averages raw EgoVLP features within step boundaries:
- Maps step times to feature frame indices
- Computes mean of overlapping frames
- Returns [num_steps, 256] array

---

## Output Format

### 1. **results.json** (Step Intervals)
```json
{
  "1_7": {
    "recording_id": "1_7",
    "steps": [
      {"step_id": 2, "start_time": 5.0, "end_time": 38.0},
      {"step_id": 0, "start_time": 55.0, "end_time": 120.0},
      {"step_id": 1, "start_time": 135.0, "end_time": 200.0}
    ],
    "embeddings_shape": [3, 256]
  }
}
```

### 2. **embeddings.npz** (Step-Level Embeddings)
```python
data = np.load('embeddings.npz')
emb_1_7 = data['1_7']  # shape: [3, 256] — 3 steps, each 256-dim
```

---

## Configuration Options

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--feature_dir` | `video_features` | Directory with `.npz` feature files |
| `--ckpt` | `pretrained/hiero_egovlp.pth` | Model checkpoint path |
| `--depth` | `2` | Decoder level to extract (0=finest, 2=~20s segments) |
| `--num_steps` | `None` | Fix step count (if not set, sample per-video) |
| `--steps_config` | `None` | JSON file for per-video step counts |
| `--temp` | `0.5` | Temperature for spectral clustering (lower = sharper) |
| `--smooth_window` | `5` | Mode-filter window for temporal smoothing |
| `--no_background` | `False` | Disable background cluster detection |
| `--use_proj_head` | `False` | Use language-aligned projection head |
| `--output` | `results.json` | Output path for step intervals |
| `--embeddings_output` | `embeddings.npz` | Output path for step embeddings |

---

## Example Usage

### Basic: Fixed Number of Steps
```bash
python infer_steps.py --num_steps 7 --feature_dir video_features
```
Identifies exactly 7 steps in each video.

### Per-Video Configuration
Create `steps.json`:
```json
{
  "1_10_360p_224.mp4_1s_1s.npz": 5,
  "1_14_360p_224.mp4_1s_1s.npz": 9
}
```
Then run:
```bash
python infer_steps.py --steps_config steps.json --feature_dir video_features
```

### Sampled from Distribution (Dataset-Realistic)
```bash
python infer_steps.py --feature_dir video_features
```
Each video gets a random step count sampled from the ground-truth distribution.

### Fine-Tuning Temporal Boundaries
```bash
python infer_steps.py --smooth_window 8 --temp 0.3 --feature_dir video_features
```
- Larger smooth_window = smoother step boundaries
- Lower temp = tighter clusters (sharper transitions)

---

## Data Flow Summary

```
video_features/
   ├─ 1_7_360p_224.mp4_1s_1s.npz  [N=180, 256]
   └─ 1_10_360p_224.mp4_1s_1s.npz [N=240, 256]
            ↓
    Load EgoVLP Features
            ↓
    HiERO Hierarchical Decoder
    (depth=2: N → M segments)
            ↓
    Spectral Clustering
    (M segments → num_steps clusters)
            ↓
    Background Detection
    (identify noise cluster)
            ↓
    Temporal Smoothing + Merging
    (cluster labels → step intervals)
            ↓
    Step Embeddings
    (average raw features per step)
            ↓
    results.json + embeddings.npz
```

---

## Summary

**`infer_steps.py` performs a 3-stage process:**

1. **Feature Extraction**: HiERO hierarchical decoder processes raw EgoVLP features into temporally-coherent segments
2. **Clustering**: Spectral clustering groups segments into procedure steps, with optional background detection
3. **Refinement**: Temporal smoothing, fragmentation merging, and gap constraints convert cluster labels into clean step intervals

The output is both human-interpretable (step timing in JSON) and machine-readable (averaged embeddings for downstream tasks).
