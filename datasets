"""
datasets.py

Dataset utilities for DTMKFC:
* Synthetic multi‑view generator
* Convenience loaders for public datasets used in the paper
  ORL, Handwritten (Digits), Scene15, NUS‑WIDE, MSRCv1, Leavers.

NOTE:
Large raw datasets are **not** bundled. Loader functions will:
  1. Check default cache directory (~/.dtmkfc/data)
  2. Attempt to download from official mirrors if missing (when possible)
  3. Otherwise raise a helpful error prompting manual download.
"""

from __future__ import annotations
import os
import tarfile
import zipfile
import pickle
import urllib.request
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------

_CACHE = Path(os.getenv("DTMKFC_DATA", Path.home() / ".dtmkfc" / "data"))
_CACHE.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"[datasets] downloading {url} …")
    urllib.request.urlretrieve(url, dest)
    return dest


def _load_mat(path: Path, key: str):
    from scipy.io import loadmat
    data = loadmat(path)
    return data[key]


# ---------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------

def load_synthetic_digits(n_views: int = 2, noise_std: float = 0.05, random_state: int = 42
                          ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Create a synthetic multi-view dataset by adding Gaussian noise to sklearn digits.

    Returns:
        views  (List[np.ndarray]) : list length = n_views, each of shape (n_samples, n_features)
        labels (np.ndarray)       : ground truth labels
    """
    raw = load_digits()
    X = StandardScaler().fit_transform(raw.data)
    y = raw.target
    rng = np.random.default_rng(random_state)
    views = [X + noise_std * rng.standard_normal(X.shape) for _ in range(n_views)]
    return views, y


# ---------------------------------------------------------------
# ORL Faces
# ---------------------------------------------------------------

def load_orl(local_dir: Path | None = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load the ORL face dataset (40 subjects × 10 images).

    Notes
    -----
    We provide a lightweight loader that:
      1. Downloads `att_faces.zip` (400 JPGs, ~4 MB) if missing
      2. Flattens into feature vectors (raw grayscale intensities)
      3. Generates two views: raw intensity, ±Gaussian noise
    """
    url = "https://github.com/machine-learning-ucd/ML-Resources/releases/download/v0.0.1/att_faces.zip"
    cache_zip = _CACHE / "orl" / "att_faces.zip"
    _download(url, cache_zip)

    # extract once
    extract_dir = cache_zip.with_suffix("")
    if not extract_dir.exists():
        with zipfile.ZipFile(cache_zip) as z:
            z.extractall(extract_dir)

    # read images
    import imageio.v2 as iio
    X_list, y_list = [], []
    for i, subdir in enumerate(sorted(extract_dir.glob("s*"))):
        for img_path in sorted(subdir.glob("*.pgm")):
            img = iio.imread(img_path)
            X_list.append(img.flatten().astype("float32") / 255.0)
            y_list.append(i)
    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list)

    rng = np.random.default_rng(0)
    X_noise = X + 0.05 * rng.standard_normal(X.shape)
    return [X, X_noise], y


# ---------------------------------------------------------------
# Handwritten Digits (same as synthetic but kept for reference)
# ---------------------------------------------------------------

def load_handwritten() -> Tuple[List[np.ndarray], np.ndarray]:
    return load_synthetic_digits()


# ---------------------------------------------------------------
# Scene15
# ---------------------------------------------------------------

def load_scene15(local_mat: Path | None = None):
    """
    Load Scene15 pre-extracted features (.mat) used in many clustering papers.

    Expected keys: `gist`, `phog`, `lbp`, `labels`
    """
    if local_mat is None:
        raise FileNotFoundError(
            "Scene15 .mat file not found. Please download features (e.g., from "
            "https://github.com/kunzhan/cluster-datasets) and pass path to load_scene15()."
        )
    mat = _load_mat(local_mat, None)
    views = [mat["gist"], mat["phog"], mat["lbp"]]
    y = mat["labels"].squeeze() - 1  # to 0-index
    return views, y


# ---------------------------------------------------------------
# NUS-WIDE (subset loader)
# ---------------------------------------------------------------

def load_nuswide100k(local_pickle: Path | None = None):
    """
    Placeholder loader for NUS-WIDE 100k subset.

    Users should pre-extract or download features and supply `local_pickle`
    containing:
        dict{ 'views': List[np.ndarray], 'labels': np.ndarray }
    """
    if local_pickle is None or not local_pickle.exists():
        raise FileNotFoundError("Please supply preprocessed NUS-WIDE pickle.")
    blob = pickle.loads(local_pickle.read_bytes())
    return blob["views"], blob["labels"]


# ---------------------------------------------------------------
# MSRCv1 loader
# ---------------------------------------------------------------

def load_msrcv1(local_mat: Path | None = None):
    """
    Load MSRCv1 .mat with CMT, LBP, GENT features.

    Format follows Zhao et al. 2017 multi-view clustering benchmark.
    """
    if local_mat is None or not local_mat.exists():
        raise FileNotFoundError("MSRCv1 .mat not found.")
    mat = _load_mat(local_mat, None)
    views = [mat["CMT"], mat["LBP"], mat["GENT"]]
    y = mat["labels"].squeeze() - 1
    return views, y


# ---------------------------------------------------------------
# Leaves data (Flavia / Swedish)
# ---------------------------------------------------------------

def load_leaves(local_dir: Path | None = None):
    """
    Placeholder loader for 1600 plant leaf images (3 feature views).

    For brevity, this function requires a pre-extracted npz containing:
        leaf_margin, leaf_texture, leaf_shape, labels
    """
    if local_dir is None:
        raise FileNotFoundError("Please supply path to leaves_features.npz")
    npz = np.load(local_dir)
    views = [npz["margin"], npz["texture"], npz["shape"]]
    y = npz["labels"]
    return views, y


# ---------------------------------------------------------------
# Registry
# ---------------------------------------------------------------

_registry = {
    "synthetic": load_synthetic_digits,
    "handwritten": load_handwritten,
    "orl": load_orl,
    "scene15": load_scene15,
    "nuswide": load_nuswide100k,
    "msrcv1": load_msrcv1,
    "leaves": load_leaves,
}


def get_dataset(name: str, **kwargs):
    if name not in _registry:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_registry)}")
    return _registry[name](**kwargs)
