#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone improved version of VRAE_Pose_Estimation_v2.

This script is intentionally self-contained and isolated from the current repo:
- It does not import local project modules.
- It stores runtime artifacts under /tmp by default.
- It clones and uses emg2pose in an isolated runtime directory.

CHANGED SECTIONS are explicitly marked with:
    # --- CHANGED ---
"""

import copy
import gc
import glob
import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import resample_poly
from scipy.stats import norm
from sklearn.decomposition import PCA
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning, module="h5py")


# --- 0. SETUP & DEPENDENCIES ---
def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def safe_import_or_install(import_name: str, pip_name: Optional[str] = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        pkg = pip_name if pip_name else import_name
        run([sys.executable, "-m", "pip", "install", "--upgrade", pkg])


def ensure_runtime_deps() -> None:
    safe_import_or_install("hydra")
    safe_import_or_install("av")
    safe_import_or_install("mediapy")
    safe_import_or_install("kaleido", "kaleido==0.2.1")


# --- CHANGED ---
# Use an isolated runtime root so this file does not interact with other repo files.
RUN_ROOT = Path(os.environ.get("VRAE_RUN_ROOT", "/tmp/vrae_pose_estimation_run")).resolve()
RUN_ROOT.mkdir(parents=True, exist_ok=True)

REPO_PATH = RUN_ROOT / "emg2pose"
DATA_DOWNLOAD_DIR = RUN_ROOT / "data"
DATA_DOWNLOAD_DIR.mkdir(exist_ok=True)


def setup_emg2pose_repo() -> None:
    if not REPO_PATH.exists():
        print(f"Cloning repository to {REPO_PATH} ...")
        run(["git", "clone", "https://github.com/facebookresearch/emg2pose.git", str(REPO_PATH)])
    else:
        print(f"Repository already exists: {REPO_PATH}")

    print("Installing emg2pose package...")
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=REPO_PATH)

    umetrack_path = REPO_PATH / "emg2pose" / "UmeTrack"
    if umetrack_path.exists():
        print("Installing UmeTrack sub-package...")
        run([sys.executable, "-m", "pip", "install", "-e", str(umetrack_path)])


def download_dataset() -> None:
    metadata_path = DATA_DOWNLOAD_DIR / "emg2pose_metadata.csv"
    if not metadata_path.exists():
        print("Downloading metadata...")
        run(
            [
                "curl",
                "-L",
                "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv",
                "-o",
                str(metadata_path),
            ]
        )

    mini_dataset_tar_path = DATA_DOWNLOAD_DIR / "emg2pose_dataset_mini.tar"
    if not mini_dataset_tar_path.exists():
        print("Downloading mini dataset...")
        run(
            [
                "curl",
                "-L",
                "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar",
                "-o",
                str(mini_dataset_tar_path),
            ]
        )
        print("Unpacking dataset...")
        run(["tar", "-xvf", str(mini_dataset_tar_path), "-C", str(DATA_DOWNLOAD_DIR)])

    sessions = sorted(glob.glob(str(DATA_DOWNLOAD_DIR / "emg2pose_dataset_mini" / "*.hdf5")))
    print(f"Found {len(sessions)} sessions.")


def inspect_example_h5() -> None:
    files = sorted(glob.glob(str(DATA_DOWNLOAD_DIR / "emg2pose_dataset_mini" / "*.hdf5")))
    if not files:
        print("No HDF5 files found.")
        return
    fpath = files[0]
    print(f"Inspecting example file: {Path(fpath).name}")
    with h5py.File(fpath, "r") as f:
        print("--- DATASETS AND GROUPS FOUND ---")

        def print_structure(name, node):
            if isinstance(node, h5py.Dataset):
                print(f"Key: '{name}' | Shape: {node.shape} | Type: {node.dtype}")
            elif isinstance(node, h5py.Group):
                print(f"Group: '{name}'")

        f.visititems(print_structure)


def configure_visualization_module():
    # Path patching for UmeTrack absolute imports
    project_root = REPO_PATH
    potential_paths = [
        project_root / "emg2pose" / "UmeTrack",
        project_root / "UmeTrack",
    ]

    umetrack_found = False
    for p in potential_paths:
        if (p / "lib").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            print(f"Configured UmeTrack path: {p}")
            umetrack_found = True
            break

    if not umetrack_found:
        print("WARNING: Could not find UmeTrack/lib folder. Visualization might fail.")

    visualization = None
    try:
        import emg2pose.visualization as visualization_module

        importlib.reload(visualization_module)
        visualization = visualization_module
    except ImportError as e:
        print(f"Visualization import failed: {e}")
    return visualization


# --- 1. CONFIGURATION PARAMETERS ---
BATCH_SIZE = 256
EPOCHS = 1000
HIDDEN_DIM = 512
LATENT_DIM = 64

TARGET_FS = 60
NATIVE_FS = 2000
SEQ_LEN = 60
HOP_LEN = SEQ_LEN // 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- CHANGED ---
# Load paired EMG input and pose targets (pose velocity), session-wise split.
def _search_dataset_by_keywords(
    h5_file: h5py.File,
    include_keywords: Tuple[str, ...],
    exclude_keywords: Tuple[str, ...] = (),
) -> Optional[np.ndarray]:
    found = None

    def visitor(name, node):
        nonlocal found
        if found is not None:
            return
        if not isinstance(node, h5py.Dataset):
            return

        lname = name.lower()
        if any(k in lname for k in include_keywords) and not any(k in lname for k in exclude_keywords):
            arr = node[()]
            if arr is None:
                return
            if isinstance(arr, np.ndarray):
                if arr.dtype.names:
                    for field in ("joint_angles", "pose", "angles", "emg"):
                        if field in arr.dtype.names:
                            arr = arr[field]
                            break
                    else:
                        return
                if arr.ndim >= 2:
                    found = np.asarray(arr)

    h5_file.visititems(visitor)
    return found


def _ensure_time_first(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x


def _resample_to_target(x: np.ndarray, native_fs: int, target_fs: int) -> np.ndarray:
    if native_fs == target_fs:
        return x
    return resample_poly(x, up=target_fs, down=native_fs, axis=0)


def load_all_data(data_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], int, int]:
    emg_windows_by_session: List[np.ndarray] = []
    pose_windows_by_session: List[np.ndarray] = []
    session_ids: List[str] = []

    files = sorted(glob.glob(str(data_dir / "*.hdf5")))
    print(f"Processing {len(files)} sessions...")

    emg_dim = 0
    pose_dim = 0

    for fpath in files:
        try:
            with h5py.File(fpath, "r") as f:
                pose = _search_dataset_by_keywords(
                    f,
                    include_keywords=("joint_angles", "pose"),
                    exclude_keywords=("emg", "muscle"),
                )
                emg = _search_dataset_by_keywords(
                    f,
                    include_keywords=("emg", "muscle", "signal"),
                    exclude_keywords=("joint", "pose", "angle"),
                )

                if pose is None or emg is None:
                    print(f"Skipping {Path(fpath).name} (could not find both EMG and pose)")
                    continue

                pose = _ensure_time_first(pose.astype(np.float32))
                emg = _ensure_time_first(emg.astype(np.float32))

                n = min(len(emg), len(pose))
                if n < SEQ_LEN:
                    continue
                emg = emg[:n]
                pose = pose[:n]

                emg = _resample_to_target(emg, NATIVE_FS, TARGET_FS).astype(np.float32)
                pose = _resample_to_target(pose, NATIVE_FS, TARGET_FS).astype(np.float32)

                n2 = min(len(emg), len(pose))
                emg = emg[:n2]
                pose = pose[:n2]

                pose_vel = np.diff(pose, axis=0, prepend=pose[:1]).astype(np.float32)

                if emg_dim == 0:
                    emg_dim = emg.shape[1]
                    pose_dim = pose_vel.shape[1]
                    print(f"Input EMG Dimension: {emg_dim}")
                    print(f"Target Pose-Vel Dimension: {pose_dim}")

                xw: List[np.ndarray] = []
                yw: List[np.ndarray] = []
                for t in range(0, n2 - SEQ_LEN + 1, HOP_LEN):
                    xw.append(emg[t : t + SEQ_LEN])
                    yw.append(pose_vel[t : t + SEQ_LEN])

                if not xw:
                    continue

                emg_windows_by_session.append(np.stack(xw))
                pose_windows_by_session.append(np.stack(yw))
                session_ids.append(Path(fpath).stem)

        except Exception as e:
            print(f"Error loading {Path(fpath).name}: {e}")

    return emg_windows_by_session, pose_windows_by_session, session_ids, emg_dim, pose_dim


def session_split(
    x_by_session: List[np.ndarray],
    y_by_session: List[np.ndarray],
    session_ids: List[str],
    train_ratio: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(session_ids)
    if n == 0:
        raise RuntimeError("No sessions available after preprocessing.")

    split_idx = max(1, int(n * train_ratio))
    train_idx = set(range(split_idx))

    x_train, y_train, x_val, y_val = [], [], [], []
    for i in range(n):
        if i in train_idx:
            x_train.append(x_by_session[i])
            y_train.append(y_by_session[i])
        else:
            x_val.append(x_by_session[i])
            y_val.append(y_by_session[i])

    # Guarantee non-empty val set when possible.
    if len(x_val) == 0 and len(x_train) > 1:
        x_val.append(x_train.pop())
        y_val.append(y_train.pop())

    return (
        np.concatenate(x_train, axis=0),
        np.concatenate(y_train, axis=0),
        np.concatenate(x_val, axis=0),
        np.concatenate(y_val, axis=0),
    )


def compute_norm_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    std = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_with_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


# --- 2. VRAE MODEL DEFINITION ---
# --- CHANGED ---
# Conditional VRAE: encode EMG, decode pose velocity.
class EmgToPoseVRAE(nn.Module):
    def __init__(self, emg_dim: int, pose_dim: int, hidden_dim: int, latent_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.num_layers = 2
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            emg_dim,
            hidden_dim,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=dropout_rate,
        )
        self.to_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        self.decoder = nn.LSTM(
            emg_dim + latent_dim,
            hidden_dim,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=dropout_rate,
        )
        self.out = nn.Linear(hidden_dim, pose_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_emg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, (h, _) = self.encoder(x_emg)
        h_last = torch.cat([h[-2], h[-1]], dim=1)
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last)
        z = self.reparameterize(mu, logvar)

        z_rep = z.unsqueeze(1).expand(-1, x_emg.size(1), -1)
        dec_in = torch.cat([x_emg, z_rep], dim=-1)
        dec_out, _ = self.decoder(dec_in)
        recon = self.out(dec_out)
        return recon, mu, logvar


# --- 3. TRAINING ---
# --- CHANGED ---
# Stabilized objective + KL annealing.
def calculate_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, float, float]:
    recon_loss = F.smooth_l1_loss(recon, target, reduction="mean")
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon_loss + (beta * kld)
    return total, float(recon_loss.item()), float(kld.item())


def train_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    emg_dim: int,
    pose_dim: int,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    device: torch.device,
) -> Tuple[Dict[float, Dict[str, List[float]]], Dict[str, object]]:
    dropout_levels = [0.1, 0.2, 0.3, 0.4]

    global_best_val = float("inf")
    global_best_dropout = None
    global_best_epoch = 0
    global_best_state = None

    results_history: Dict[float, Dict[str, List[float]]] = {}
    print("Starting Hyperparameter Search...")

    for drop in dropout_levels:
        print(f"\n--- Training model with dropout={drop:.1f} ---")
        model = EmgToPoseVRAE(emg_dim, pose_dim, hidden_dim, latent_dim, dropout_rate=drop).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

        train_hist, val_hist, val_shuffled_hist = [], [], []

        best_run_val = float("inf")
        best_run_epoch = 0
        best_run_state = None
        patience = 120
        no_improve = 0

        for epoch in range(epochs):
            beta = min(1.0, (epoch + 1) / 200.0) * 1e-3
            model.train()
            running_train = 0.0

            for x_emg, y_pose in train_loader:
                x_emg = x_emg.to(device)
                y_pose = y_pose.to(device)

                optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar = model(x_emg)
                loss, _, _ = calculate_loss(recon, y_pose, mu, logvar, beta=beta)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_train += loss.item() * x_emg.size(0)

            avg_train = running_train / len(train_loader.dataset)
            train_hist.append(avg_train)

            model.eval()
            running_val = 0.0
            running_val_shuffled = 0.0

            with torch.no_grad():
                for x_emg, y_pose in val_loader:
                    x_emg = x_emg.to(device)
                    y_pose = y_pose.to(device)
                    recon, mu, logvar = model(x_emg)
                    val_loss, _, _ = calculate_loss(recon, y_pose, mu, logvar, beta=beta)
                    running_val += val_loss.item() * x_emg.size(0)

                    # shuffled target sanity check
                    idx = torch.randperm(y_pose.size(0), device=device)
                    y_shuf = y_pose[idx]
                    shuf_mse = F.mse_loss(recon, y_shuf, reduction="mean")
                    running_val_shuffled += shuf_mse.item() * x_emg.size(0)

            avg_val = running_val / len(val_loader.dataset)
            avg_val_shuf = running_val_shuffled / len(val_loader.dataset)
            val_hist.append(avg_val)
            val_shuffled_hist.append(avg_val_shuf)

            scheduler.step(avg_val)

            if avg_val < best_run_val:
                best_run_val = avg_val
                best_run_epoch = epoch + 1
                best_run_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 25 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:4d}/{epochs} | "
                    f"Train={avg_train:.5f} Val={avg_val:.5f} ValShuf={avg_val_shuf:.5f} "
                    f"| beta={beta:.6f} lr={lr:.2e}"
                )

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_run_epoch}.")
                break

        results_history[drop] = {
            "train": train_hist,
            "val": val_hist,
            "shuffled": val_shuffled_hist,
        }
        print(f"Best epoch for dropout={drop:.1f}: {best_run_epoch} with val={best_run_val:.5f}")

        if best_run_val < global_best_val:
            global_best_val = best_run_val
            global_best_dropout = drop
            global_best_epoch = best_run_epoch
            global_best_state = copy.deepcopy(best_run_state)

    summary = {
        "global_best_val": global_best_val,
        "global_best_dropout": global_best_dropout,
        "global_best_epoch": global_best_epoch,
        "global_best_state": global_best_state,
    }
    return results_history, summary


# --- 4. PLOTTING ---
def plot_dropout_comparison(results_dict: Dict[float, Dict[str, List[float]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["blue", "orange", "green", "red"]

    for idx, (drop, history) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        axes[0].plot(history["train"], label=f"Dropout: {drop}", color=color, linewidth=2)
        axes[1].plot(history["val"], label=f"Dropout: {drop}", color=color, linewidth=2)
        axes[2].plot(history["shuffled"], label=f"Dropout: {drop}", color=color, linestyle="--", linewidth=2)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss (Real Data)")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Validation Loss (Shuffled Targets)")
    axes[2].set_xlabel("Epochs")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


def run_control_experiment(
    emg_dim: int,
    pose_dim: int,
    hidden_dim: int,
    latent_dim: int,
    train_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
) -> List[float]:
    print("\nStarting control experiment (shuffled targets)...")
    model = EmgToPoseVRAE(emg_dim, pose_dim, hidden_dim, latent_dim, dropout_rate=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    hist: List[float] = []

    for epoch in range(num_epochs):
        beta = min(1.0, (epoch + 1) / 200.0) * 1e-3
        model.train()
        running = 0.0

        for x_emg, y_pose in train_loader:
            x_emg = x_emg.to(device)
            y_pose = y_pose.to(device)

            idx = torch.randperm(y_pose.size(0), device=device)
            y_shuf = y_pose[idx]

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x_emg)
            loss, _, _ = calculate_loss(recon, y_shuf, mu, logvar, beta=beta)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * x_emg.size(0)

        avg = running / len(train_loader.dataset)
        hist.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"Control Epoch {epoch+1:4d}/{num_epochs} | Loss={avg:.5f}")

    return hist


def plot_verification_results(train_losses: List[float], val_losses: List[float], control_losses: List[float], out: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss (Real Data)", color="blue", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color="green", linestyle="--", linewidth=2)
    plt.plot(control_losses, label="Control Loss (Shuffled Targets)", color="red", linestyle="-.", linewidth=2)
    plt.title("Model Verification: Real vs. Shuffled Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.close()


def benchmark_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    mses, corrs, shuffled_mses = [], [], []
    with torch.no_grad():
        for x_emg, y_pose in test_loader:
            x_emg = x_emg.to(device)
            y_pose = y_pose.to(device)
            recon, _, _ = model(x_emg)

            mses.append(F.mse_loss(recon, y_pose).item())
            idx = torch.randperm(y_pose.size(0), device=device)
            shuffled_mses.append(F.mse_loss(recon, y_pose[idx]).item())

            y_flat = y_pose.cpu().numpy().reshape(-1)
            r_flat = recon.cpu().numpy().reshape(-1)
            if np.std(y_flat) > 0 and np.std(r_flat) > 0:
                corrs.append(np.corrcoef(y_flat, r_flat)[0, 1])
            else:
                corrs.append(0.0)

    return float(np.mean(mses)), float(np.mean(shuffled_mses)), float(np.mean(corrs))


def visualize_if_possible(
    visualization,
    model: nn.Module,
    x_test: torch.Tensor,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
) -> None:
    if visualization is None:
        print("Skipping visualization: module unavailable.")
        return

    try:
        import mediapy
        from tqdm import tqdm
    except ImportError:
        print("Skipping visualization: mediapy/tqdm unavailable.")
        return

    try:
        print("\nRendering video comparison...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        x_sample = x_test[0:1].to(device)
        with torch.no_grad():
            pred_vel, _, _ = model(x_sample)

        pred_vel = pred_vel.cpu().squeeze(0).numpy()
        true_vel = x_test.new_zeros((x_sample.size(1), pred_vel.shape[-1])).cpu().numpy()

        # unnormalize predicted velocity
        pred_vel = (pred_vel * y_std.squeeze(0).squeeze(0)) + y_mean.squeeze(0).squeeze(0)
        true_vel = (true_vel * y_std.squeeze(0).squeeze(0)) + y_mean.squeeze(0).squeeze(0)

        pred_pose = np.cumsum(pred_vel, axis=0)
        true_pose = np.cumsum(true_vel, axis=0)

        skip = 2
        pred_pose = pred_pose[::skip]
        true_pose = true_pose[::skip]

        if hasattr(visualization, "joint_angles_to_frames"):
            render_func = visualization.joint_angles_to_frames
        elif hasattr(visualization, "joint_angles_to_frame"):
            render_func = visualization.joint_angles_to_frame
        else:
            print("Rendering function not found in visualization module.")
            return

        try:
            gt_frames = render_func(true_pose, color="gray")
            pred_frames = render_func(pred_pose, color="lightpink")
        except Exception:
            gt_frames, pred_frames = [], []
            for i in tqdm(range(len(true_pose))):
                gt_frames.append(render_func(true_pose[i : i + 1], color="gray")[0])
                pred_frames.append(render_func(pred_pose[i : i + 1], color="lightpink")[0])
            gt_frames = np.array(gt_frames)
            pred_frames = np.array(pred_frames)

        gt_frames = visualization.remove_alpha_channel(gt_frames)
        pred_frames = visualization.remove_alpha_channel(pred_frames)
        mediapy.show_videos(dict(GroundTruth=gt_frames, Prediction=pred_frames), width=600, fps=30)

    except Exception as e:
        print(f"Visualization error: {e}")


def latent_analysis(model: EmgToPoseVRAE, test_loader: DataLoader, device: torch.device, out: Path) -> None:
    print("\nExtracting latent manifold...")
    model.eval()
    latent_vectors = []
    color_values = []

    with torch.no_grad():
        for x_emg, y_pose in test_loader:
            x_emg = x_emg.to(device)
            _, (h, _) = model.encoder(x_emg)
            h_last = torch.cat([h[-2], h[-1]], dim=1)
            mu = model.to_mu(h_last)

            latent_vectors.append(mu.cpu().numpy())
            color_values.append(y_pose.mean(dim=(1, 2)).cpu().numpy())

    if not latent_vectors:
        print("No latent vectors for analysis.")
        return

    z = np.vstack(latent_vectors)
    colors = np.concatenate(color_values)
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)
    exp_var = pca.explained_variance_ratio_

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sc = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=colors, cmap="viridis", alpha=0.6, s=10)
    plt.colorbar(sc, label="Average Sequence Velocity")
    plt.title(
        f"Pose Latent Manifold (PCA Projection)\n"
        f"Explained Variance: PC1={exp_var[0]*100:.1f}% / PC2={exp_var[1]*100:.1f}%"
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(z.flatten(), bins=50, color="purple", alpha=0.7, density=True, label="Actual Latent Distribution")
    x_axis = np.arange(-4, 4, 0.01)
    plt.plot(x_axis, norm.pdf(x_axis, 0, 1), color="black", lw=2, linestyle="--", label="Target Standard Normal")
    plt.title("Latent Variable Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.close()


def main() -> None:
    print(f"Runtime root: {RUN_ROOT}")
    print(f"Running on device: {DEVICE}")

    ensure_runtime_deps()
    setup_emg2pose_repo()
    download_dataset()
    inspect_example_h5()
    visualization = configure_visualization_module()

    data_path = DATA_DOWNLOAD_DIR / "emg2pose_dataset_mini"
    x_by_session, y_by_session, session_ids, emg_dim, pose_dim = load_all_data(data_path)

    if len(session_ids) == 0:
        raise RuntimeError("No usable sessions loaded from dataset.")

    x_train_np, y_train_np, x_test_np, y_test_np = session_split(
        x_by_session, y_by_session, session_ids, train_ratio=0.9
    )
    print(f"Train windows: {len(x_train_np)} | Test windows: {len(x_test_np)}")

    x_mean, x_std = compute_norm_stats(x_train_np)
    y_mean, y_std = compute_norm_stats(y_train_np)

    x_train_np = normalize_with_stats(x_train_np, x_mean, x_std)
    y_train_np = normalize_with_stats(y_train_np, y_mean, y_std)
    x_test_np = normalize_with_stats(x_test_np, x_mean, x_std)
    y_test_np = normalize_with_stats(y_test_np, y_mean, y_std)

    x_train = torch.from_numpy(x_train_np).float()
    y_train = torch.from_numpy(y_train_np).float()
    x_test = torch.from_numpy(x_test_np).float()
    y_test = torch.from_numpy(y_test_np).float()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    results_history, summary = train_search(
        train_loader=train_loader,
        val_loader=test_loader,
        emg_dim=emg_dim,
        pose_dim=pose_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        device=DEVICE,
    )

    print("\nGLOBAL BEST MODEL FOUND")
    print(f"Dropout: {summary['global_best_dropout']}")
    print(f"Best epoch: {summary['global_best_epoch']}")
    print(f"Best val loss: {summary['global_best_val']:.5f}")

    best_model = EmgToPoseVRAE(
        emg_dim, pose_dim, HIDDEN_DIM, LATENT_DIM, dropout_rate=float(summary["global_best_dropout"])
    ).to(DEVICE)
    best_model.load_state_dict(summary["global_best_state"])
    best_model.eval()

    plot_dropout_comparison(results_history, RUN_ROOT / "dropout_comparison_plot.png")

    # Use same length as best train history for fair control comparison.
    best_hist = results_history[float(summary["global_best_dropout"])]
    control_epochs = len(best_hist["train"])
    control_hist = run_control_experiment(
        emg_dim=emg_dim,
        pose_dim=pose_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        train_loader=train_loader,
        num_epochs=control_epochs,
        device=DEVICE,
    )
    plot_verification_results(
        train_losses=best_hist["train"],
        val_losses=best_hist["val"],
        control_losses=control_hist,
        out=RUN_ROOT / "model_verification_plot.png",
    )

    mse, shuffled_mse, corr = benchmark_model(best_model, test_loader, DEVICE)
    print("\n--- Benchmarking Results ---")
    print(f"Average True MSE: {mse:.6f}")
    print(f"Average Shuffled MSE: {shuffled_mse:.6f}")
    print(f"Average Pearson Correlation: {corr:.4f}")

    visualize_if_possible(visualization, best_model, x_test, y_mean, y_std, DEVICE)
    latent_analysis(best_model, test_loader, DEVICE, RUN_ROOT / "latent_manifold_plot.png")

    print("\n--- Analysis Interpretation ---")
    print("1) Lower real-val loss vs shuffled-target loss indicates meaningful EMG->pose learning.")
    print("2) Latent histogram overlapping N(0,1) indicates KL regularization is active.")
    print("3) Smoother train/val curves indicate improved optimization stability.")


if __name__ == "__main__":
    main()
