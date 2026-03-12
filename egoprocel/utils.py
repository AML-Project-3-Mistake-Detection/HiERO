import cv2
import torch
from torch.nn import functional as F

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def clusterize(features: torch.Tensor, n: int, temp=0.05):
    """
    Perform temporally-constrained agglomerative clustering on the given features.

    A chain connectivity matrix restricts merges to temporally adjacent segments,
    so every resulting cluster is a contiguous temporal block.  This prevents the
    merging / splitting artefacts that arise when order-agnostic spectral clustering
    assigns the same label to segments scattered across time.

    Args:
        features (torch.Tensor): The input features to cluster, expected to be a 2D tensor.
        n (int): The number of clusters to form.
        temp (float, optional): Unused; kept for API compatibility.
    Returns:
        numpy.ndarray: An array of cluster labels for each feature.
    """
    feats = F.normalize(features, p=2, dim=-1).detach().cpu().numpy()
    N = len(feats)

    if N <= n:
        # Degenerate case: each segment is its own cluster
        return np.arange(N, dtype=np.intp)

    # Chain graph: every node is only connected to its immediate temporal neighbour.
    # AgglomerativeClustering with this connectivity can only ever merge adjacent
    # segments, so the output is always a set of contiguous temporal intervals.
    chain = kneighbors_graph(
        np.arange(N, dtype=np.float32).reshape(-1, 1),
        n_neighbors=1,
        mode="connectivity",
        include_self=False,
    )

    return AgglomerativeClustering(
        n_clusters=n,
        metric="cosine",
        linkage="average",
        connectivity=chain,
    ).fit_predict(feats)


def get_fps(video_path: str) -> float:
    """
    Get the frames per second of a video.
    Args:
        video_path (str): The path to the video.
    Returns:
        int: The frames per second of the video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)

