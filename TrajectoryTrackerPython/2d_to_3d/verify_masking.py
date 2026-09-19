"""
Verifies that TrajectoryReconstructionNetwork's masking/packing logic gives every real timestep
the same prediction whether or not it's sitting inside a padded, mixed-length batch, per the
verification plan for the masking work. Run directly: `python verify_masking.py`.
"""
import numpy as np
import torch

import settings
from model import TrajectoryReconstructionNetwork


def build_model(seed=0):
    torch.manual_seed(seed)

    intrinsics = np.array([
        [800.0, 0.0, 480.0],
        [0.0, 800.0, 270.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    theta = 0.3
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    extrinsics = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_t, -sin_t, -5.0],
        [0.0, sin_t, cos_t, 15.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    model = TrajectoryReconstructionNetwork()
    model.eval()

    intrinsics = torch.from_numpy(intrinsics).to(dtype=settings.torch_dtype) # (3, 3)
    extrinsics = torch.from_numpy(extrinsics).to(dtype=settings.torch_dtype) # (4, 4)
    return model, intrinsics, extrinsics


def _expand_camera(intrinsics, extrinsics, n):
    """Repeats a single (3, 3) / (4, 4) camera config into a batch of size n, matching how a
    single-camera batch would arrive from the real dataset/collate pipeline."""
    return intrinsics.unsqueeze(0).expand(n, -1, -1), extrinsics.unsqueeze(0).expand(n, -1, -1)


def _diameters(n):
    """A plausible batch of apparent diameters, near the measured mean of ~8 px. Ignored unless
    settings.estimate_height_from_diameter is enabled, but passed either way so this suite
    exercises the same call signature the trainer uses."""
    return torch.full((n, 2), 8.0, dtype=settings.torch_dtype)


def random_trajectory(rng, length):
    xs = rng.uniform(150.0, 800.0, size=length).astype(np.float64)
    ys = rng.uniform(80.0, 450.0, size=length).astype(np.float64)
    return torch.from_numpy(np.stack([xs, ys], axis=-1)).to(dtype=settings.torch_dtype)


def pad_batch(sequences):
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.zeros((len(sequences), max_len, sequences[0].shape[-1]), dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]] = seq
    return padded, lengths


def assert_no_nan_inf(name, tensor):
    assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"


def check_batch_matches_standalone(model, intrinsics, extrinsics, lengths_list, label, rtol=1e-3, atol=1e-4):
    rng = np.random.default_rng(42)
    sequences = [random_trajectory(rng, l) for l in lengths_list]

    standalone_coords, standalone_eps = [], []
    with torch.no_grad():
        for seq in sequences:
            x = seq.unsqueeze(0) # (1, L, 2)
            lengths = torch.tensor([seq.shape[0]], dtype=torch.long)
            cam_intrinsics, cam_extrinsics = _expand_camera(intrinsics, extrinsics, 1)
            coords, eps = model(x, lengths, cam_intrinsics, cam_extrinsics, _diameters(1))
            standalone_coords.append(coords[0])
            standalone_eps.append(eps[0])

    padded_x, lengths = pad_batch(sequences)
    batch_intrinsics, batch_extrinsics = _expand_camera(intrinsics, extrinsics, len(sequences))
    with torch.no_grad():
        batch_coords, batch_eps = model(padded_x, lengths, batch_intrinsics, batch_extrinsics, _diameters(len(sequences)))

    assert_no_nan_inf(f"{label}: batch final_coords", batch_coords)
    assert_no_nan_inf(f"{label}: batch epsilons", batch_eps)

    max_len = padded_x.shape[1]
    for i, l in enumerate(lengths_list):
        real_coords = batch_coords[i, :l]
        real_eps = batch_eps[i, :l]
        torch.testing.assert_close(real_coords, standalone_coords[i], rtol=rtol, atol=atol,
                                    msg=f"{label}: sequence {i} (length {l}) final_coords mismatch between standalone and padded-batch runs")
        torch.testing.assert_close(real_eps, standalone_eps[i], rtol=rtol, atol=atol,
                                    msg=f"{label}: sequence {i} (length {l}) epsilons mismatch between standalone and padded-batch runs")

        if l < max_len:
            pad_coords = batch_coords[i, l:]
            pad_eps = batch_eps[i, l:]
            assert torch.all(pad_coords == 0), f"{label}: sequence {i} padded final_coords positions should be exactly zero"
            assert torch.all(pad_eps == 0), f"{label}: sequence {i} padded epsilons positions should be exactly zero"

    print(f"[PASS] {label}: lengths={lengths_list}")


def check_gradients_finite(model, intrinsics, extrinsics):
    rng = np.random.default_rng(7)
    sequences = [random_trajectory(rng, l) for l in [4, 11, 2]]
    padded_x, lengths = pad_batch(sequences)
    batch_intrinsics, batch_extrinsics = _expand_camera(intrinsics, extrinsics, len(sequences))

    model.train()
    coords, eps = model(padded_x, lengths, batch_intrinsics, batch_extrinsics, _diameters(len(sequences)))
    mask = torch.arange(padded_x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    loss = coords[mask].pow(2).sum() + eps[mask].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"
    model.eval()
    print("[PASS] gradients: all finite after backward on a padded, mixed-length batch")


if __name__ == "__main__":
    model, intrinsics, extrinsics = build_model()

    check_batch_matches_standalone(model, intrinsics, extrinsics, [5, 8, 3], "mixed lengths")
    check_batch_matches_standalone(model, intrinsics, extrinsics, [6, 6, 6], "equal lengths")
    check_batch_matches_standalone(model, intrinsics, extrinsics, [1, 6], "length-one sequence")
    check_batch_matches_standalone(model, intrinsics, extrinsics, [3, 60], "extreme length disparity")

    check_gradients_finite(model, intrinsics, extrinsics)

    print("\nAll masking verification checks passed.")
