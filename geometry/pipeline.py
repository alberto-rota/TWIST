import random
import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# NOTE: ``moge`` is imported lazily inside ``GeometryPipeline.__init__`` (only on
# the MoGe branch) so this module can be imported on machines without ``moge``
# installed (e.g. the CPU login node) and so the pure novel-view warping path in
# ``dataset.pseudo_gt`` — which needs only ``geometry.projections`` /
# ``geometry.transforms`` — never pulls in the heavy MoGe dependency tree.


# ---------------------------
# helper: smoothstep
def _smoothstep(x, e0, e1):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


@torch.no_grad()
def add_geometric_roughness_torch(
    normals: torch.Tensor,  # [B,3,H,W], in [-1,1] or [0,1]
    # --- blob controls ---
    n_blobs: int = 16,  # number of blobs
    avg_blob_size: float = 0.10,  # avg diameter (fraction of min(H,W) or pixels)
    size_unit: str = "fraction",
    size_spread: float = 0.6,  # lognormal spread; >0 => many small blobs
    elongation_bias: float = 0.6,  # 0: circular, 1: very elongated on avg
    falloff_mean: float = 10,  # mean softness (0=hard edge, 0.5=soft halo)
    falloff_jitter: float = 10,  # variation of falloff per blob
    edge_wobble: float = 0.6,  # amplitude of border perturbation (0..1)
    warp_scale: int = 20,  # spatial scale (px) of border perturbation
    min_separation: float = 0.06,  # keep centers apart (fraction of min(H,W))
    # --- micro-geometry controls (unchanged) ---
    wavelength_px: float = 12.0,
    wavelength_jitter: float = 0.5,
    orientation_anisotropy: float = 0.4,
    octaves: int = 2,
    roughness_strength: float = 10,  # average angular deviation (radians)
    # misc
    seed=None,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    B, C, H, W = normals.shape
    assert C == 3
    dev = normals.device

    g = torch.Generator(device=dev)
    g.manual_seed(random.randint(0, 1000000))
    # normalize input to [-1,1]
    n = normals.clone()
    if n.min() >= 0:
        n = n * 2 - 1.0
    n = F.normalize(n, dim=1)

    # base grids
    yy, xx = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )

    # fbm-like noise field for warping (edge perturbations)
    def _smooth_noise(scale=28):
        h0, w0 = max(2, H // scale), max(2, W // scale)
        base = torch.randn(
            1, 2, h0, w0, generator=g, device=dev
        )  # 2 channels -> 2D warp
        field = F.interpolate(base, size=(H, W), mode="bicubic", align_corners=False)
        return field  # [1,2,H,W]

    warp_field = _smooth_noise(scale=max(6, warp_scale))  # shared statistics
    warp_field = warp_field / (
        warp_field.std(dim=(-2, -1), keepdim=True) + 1e-8
    )  # normalize

    # ---------- soft, perturbed super-ellipse blobs ----------
    def make_soft_blob_mask() -> torch.Tensor:
        mask = torch.zeros(1, H, W, device=dev)
        d_mean = (
            (avg_blob_size * min(H, W))
            if size_unit == "fraction"
            else float(avg_blob_size)
        )
        d_mean = max(4.0, d_mean)
        min_sep_px = max(4.0, min_separation * min(H, W))

        centers = []
        tries = 0
        while len(centers) < n_blobs and tries < 6000:
            tries += 1
            cx = torch.empty((), device=dev).uniform_(0, W, generator=g).item()
            cy = torch.empty((), device=dev).uniform_(0, H, generator=g).item()
            if all(
                ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 >= min_sep_px
                for px, py in centers
            ):
                centers.append((cx, cy))

        for cx, cy in centers:
            # sample size from lognormal (many small, some large)
            ln_sigma = size_spread * 0.5  # softer control
            s = torch.exp(
                torch.empty((), device=dev).normal_(0, ln_sigma, generator=g)
            )  # ~lognormal
            D = d_mean * s
            # elongation: sample axis ratio biased to small values
            r = torch.clamp(
                1.0
                - torch.empty((), device=dev).uniform_(0, elongation_bias, generator=g),
                0.15,
                1.0,
            )
            a = D * 0.5  # major semi-axis
            b = D * 0.5 * r  # minor semi-axis
            theta = torch.empty((), device=dev).uniform_(0, 3.1416, generator=g)
            p = torch.empty((), device=dev).uniform_(
                1.8, 3.2, generator=g
            )  # super-ellipse exponent

            # coordinate warp for irregularity
            wob_amp = (
                edge_wobble * 0.35 * D
            )  # scale by size so small blobs aren't shredded
            X = xx + wob_amp * warp_field[:, 0] - cx
            Y = yy + wob_amp * warp_field[:, 1] - cy

            ct, st = torch.cos(theta), torch.sin(theta)
            xr = ct * X + st * Y
            yr = -st * X + ct * Y

            # super-ellipse implicit metric f= (|xr/a|^p + |yr/b|^p)
            f = torch.pow(torch.abs(xr) / (a + 1e-8), p) + torch.pow(
                torch.abs(yr) / (b + 1e-8), p
            )

            # soft falloff via smoothstep around f=1 with randomized width
            soft = (
                falloff_mean
                * (
                    1.0
                    + torch.empty((), device=dev).uniform_(
                        -falloff_jitter, falloff_jitter, generator=g
                    )
                )
            ).clamp(0.05, 0.7)
            edge0 = 1.0 - soft  # inside value to start softening
            edge1 = 1.0 + soft  # outside value where it goes to 0
            blob = (1.0 - _smoothstep(f, edge0, edge1)).clamp(
                0, 1
            )  # 1 at core -> 0 outside
            blob = blob.unsqueeze(0)

            mask = (mask + blob).clamp(0, 1)

        # mild blur to merge micro-holes but keep shapes
        mask = (
            F.gaussian_blur(mask, (5, 5), sigma=(1.2, 1.2))
            if hasattr(F, "gaussian_blur")
            else mask
        )
        return mask  # [1,H,W]

    mask = torch.cat([make_soft_blob_mask() for _ in range(B)], dim=0)  # [B,1,H,W]

    # ---------- micro-geometry (same as before) ----------
    # band-limited Gabor-like height
    yyn, xxn = torch.meshgrid(
        torch.linspace(-1, 1, H, device=dev),
        torch.linspace(-1, 1, W, device=dev),
        indexing="ij",
    )
    height = torch.zeros(B, 1, H, W, device=dev)
    for o in range(octaves):
        lam = wavelength_px * (0.5**o)
        base_freq = 1.0 / max(lam, 1.0)
        dom = torch.empty(B, device=dev).uniform_(0, 3.1416, generator=g)
        theta = dom + torch.empty(B, device=dev).uniform_(
            -3.1416 * orientation_anisotropy * 0.25,
            3.1416 * orientation_anisotropy * 0.25,
            generator=g,
        )
        freq = base_freq * torch.clamp(
            1.0
            + torch.empty(B, device=dev).uniform_(
                -wavelength_jitter, wavelength_jitter, generator=g
            ),
            0.25,
            4.0,
        )
        phase = torch.empty(B, device=dev).uniform_(0, 6.2832, generator=g)
        for b in range(B):
            ct, st = torch.cos(theta[b]), torch.sin(theta[b])
            u = ct * xxn + st * yyn
            height[b : b + 1] += torch.sin(2 * 3.1416 * freq[b] * u + phase[b]) * (
                1.0 / (2**o)
            )
    height = height - height.mean(dim=(-2, -1), keepdim=True)
    height = (
        F.gaussian_blur(height, (5, 5), sigma=(1.0, 1.0))
        if hasattr(F, "gaussian_blur")
        else height
    )
    height = height * mask

    # slopes
    kx = (
        torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=dev, dtype=torch.float32
        ).view(1, 1, 3, 3)
        / 8.0
    )
    ky = (
        torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=dev, dtype=torch.float32
        ).view(1, 1, 3, 3)
        / 8.0
    )
    dhdx = F.conv2d(height, kx, padding=1)
    dhdy = F.conv2d(height, ky, padding=1)

    # TBN from n
    ref = torch.tensor([0.0, 0.0, 1.0], device=dev).view(1, 3, 1, 1).expand(B, -1, H, W)
    parallel = torch.abs((n * ref).sum(1, keepdim=True)) > 0.99
    ref = torch.where(
        parallel, torch.tensor([0.0, 1.0, 0.0], device=dev).view(1, 3, 1, 1), ref
    )
    t = F.normalize(torch.cross(ref, n, dim=1), dim=1)
    b = F.normalize(torch.cross(n, t, dim=1), dim=1)

    # unscaled perturbation from slopes
    offset = (-dhdx) * t + (-dhdy) * b

    # scale so average deviation ≈ roughness_strength (radians)
    test = F.normalize(n + offset, dim=1)
    cosang = (n * test).sum(1).clamp(-1, 1)
    mean_angle = torch.acos(cosang).mean().detach().item() + 1e-8
    scale = roughness_strength / mean_angle
    n_pert = F.normalize(n + offset * scale, dim=1)

    # blend by soft mask
    noisy = F.normalize(torch.lerp(n, n_pert, mask), dim=1)
    return (noisy, mask) if return_mask else (noisy, None)


def time_module(module_name):
    """
    Decorator to time method execution when timing is enabled.

    Args:
        module_name: String identifier for the module being timed

    Usage:
        @time_module("depth_estimation")
        def compute_depth(self, image):
            # method implementation
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "_timing_enabled", False):
                return func(self, *args, **kwargs)

            # Ensure GPU operations are complete before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
            finally:
                # Ensure GPU operations are complete after timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                self.timing_results[module_name].append(elapsed)

            return result

        return wrapper

    return decorator


class GeometryPipeline(nn.Module):
    """
    Pipeline for computing geometry (depth, normals, intrinsics) from RGB images using MoGe, EndoSynth, or Intel models.
    
    Supports:
    - MoGe models: e.g., "Ruicheng/moge-2-vits-normal" (returns depth, normals, intrinsics)
    - EndoSynth models: e.g., "endosynth:dav2" (returns depth only, dummy normals/intrinsics)
    - Intel models: e.g., "Intel/beit-base-384" (returns depth only, dummy normals/intrinsics)
    """
    def __init__(
        self,
        geometry_model_name="Ruicheng/moge-2-vits-normal",
        height=None,
        width=None,
        enable_timing=False,
        device="cuda",
        return_normalized_depth=True,
        ):
        super().__init__()
        # If height/width are specified, use them; otherwise output will match input resolution
        self.target_height = height
        self.target_width = width
        self.device = device
        self._timing_enabled = enable_timing
        self.timing_results = defaultdict(list)
        self.return_normalized_depth = return_normalized_depth
        
        # Check if EndoSynth model is requested
        import warnings
        if geometry_model_name.lower() == "endosynth" or geometry_model_name.lower().startswith("endosynth:"):

            # Parse variant if specified (e.g., "endosynth:dav1" -> "dav1")
            if ":" in geometry_model_name:
                endosynth_variant = geometry_model_name.split(":", 1)[1]
            else:
                endosynth_variant = "dav1"  # Default to dav1

            # Add EndoSynth paths if not already there
            endosynth_base = Path(__file__).parent.parent.parent / "EndoSynth"
            endosynth_third_party = endosynth_base / "third_party"

            # Add Depth-Anything for depth_anything imports (DAv1)
            depth_anything_path = endosynth_third_party / "Depth-Anything"
            if str(depth_anything_path) not in sys.path:
                sys.path.insert(0, str(depth_anything_path))

            # Add Depth-Anything-V2 for depth_anything_v2 imports (DAv2)
            depth_anything_v2_path = endosynth_third_party / "Depth-Anything-V2"
            if str(depth_anything_v2_path) not in sys.path:
                sys.path.insert(0, str(depth_anything_v2_path))

            # Add MiDaS for midas imports
            midas_path = endosynth_third_party / "MiDaS"
            if str(midas_path) not in sys.path:
                sys.path.insert(0, str(midas_path))

            # Add EndoDAC for models.endodac imports
            endodac_path = endosynth_third_party / "EndoDAC"
            if str(endodac_path) not in sys.path:
                sys.path.insert(0, str(endodac_path))
            
            # Also add EndoSynth parent directory for EndoSynth.endosynth imports
            endosynth_parent_path = Path(__file__).parent.parent.parent
            if str(endosynth_parent_path) not in sys.path:
                sys.path.insert(0, str(endosynth_parent_path))
            
            # Import and load EndoSynth model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from EndoSynth.endosynth.models import load
                self.geometry_model = load(endosynth_variant, torch.device(device))
            self.use_endosynth = True
        else:
            # Load MoGe model for joint depth and normal estimation
            from moge.model.v2 import MoGeModel

            self.geometry_model = MoGeModel.from_pretrained(geometry_model_name)
            self.geometry_model.eval()  # Set to eval mode
            self.geometry_model.to(self.device)
            self.use_endosynth = False

    def enable_timing_mode(self, enabled=True):
        """Enable or disable timing mode."""
        self._timing_enabled = enabled

    def reset_timing_stats(self):
        """Clear all timing statistics."""
        self.timing_results.clear()

    def get_timing_stats(self, detailed=False):
        """
        Get timing statistics for all modules.

        Args:
            detailed: If True, return all measurements. If False, return summary stats.

        Returns:
            dict: Timing statistics in milliseconds
        """
        if not self.timing_results:
            return {}

        if detailed:
            return dict(self.timing_results)

        # Return summary statistics
        stats = {}
        for module_name, times in self.timing_results.items():
            times_array = np.array(times)
            stats[module_name] = {
                "mean_ms": float(np.mean(times_array)),
                "std_ms": float(np.std(times_array)),
                "min_ms": float(np.min(times_array)),
                "max_ms": float(np.max(times_array)),
                "total_ms": float(np.sum(times_array)),
                "count": len(times),
            }
        return stats

    def print_timing_stats(self):
        """Print a formatted timing report."""
        stats = self.get_timing_stats()
        if not stats:
            print("No timing data available. Enable timing mode first.")
            return

        print("\n" + "=" * 60)
        print("GeometryPipeline Timing Report")
        print("=" * 60)
        print(f"{'Module':<25} {'Mean (ms)':<10} {'Std (ms)':<10} {'Count':<8}")
        print("-" * 60)

        total_time = 0
        for module_name, module_stats in sorted(stats.items()):
            mean_time = module_stats["mean_ms"]
            std_time = module_stats["std_ms"]
            count = module_stats["count"]
            total_time += module_stats["total_ms"]

            print(f"{module_name:<25} {mean_time:<10.2f} {std_time:<10.2f} {count:<8}")

        print("-" * 60)
        print(f"{'Total time':<25} {total_time:<10.2f} ms")
        print("=" * 60)

    @time_module("geometry_estimation")
    def compute_geometry(self, image, return_normalized: Optional[bool] = None):
        """
        Compute depth, normals, and intrinsics from RGB image(s) using MoGe, EndoSynth, or Intel models.

        Args:
            image: [B,3,H,W] RGB image (0-1 normalized) OR [B,2,3,H,W] pair of images
            return_normalized: If ``None``, use ``self.return_normalized_depth``. If ``False``,
                skip min–max depth normalisation so depth stays in **model units** (needed for
                :class:`dataset.pseudo_gt.PseudoGTGenerator` metric consistency).

        Returns:
            tuple: (depth, normals, intrinsics)
                - If single image: depth [B,1,H,W], normals [B,3,H,W], intrinsics [B,3,3]
                - If image pair: depth [B,2,1,H,W], normals [B,2,3,H,W], intrinsics [B,2,3,3]
                
        Note:
            - MoGe models return depth, normals, and intrinsics
            - EndoSynth and Intel models return depth only (normals and intrinsics are dummy/estimated)
        """
        # Check if input is a pair of images [B,2,3,H,W]
        is_pair = image.ndim == 5 and image.shape[1] == 2
        
        if is_pair:
            # Reshape pair to [B*2,3,H,W] for batch processing
            B, num_images, C, H, W = image.shape
            image_flat = image.view(B * num_images, C, H, W)  # [B*2,3,H,W]
            input_H, input_W = H, W
        else:
            # Single image [B,3,H,W]
            image_flat = image
            B = image.shape[0]
            num_images = 1
            input_H, input_W = image.shape[2], image.shape[3]

        # Determine target resolution: use input image dimensions by default, or constructor params if specified
        if self.target_height is not None and self.target_width is not None:
            target_H, target_W = self.target_height, self.target_width
        else:
            # Use input image dimensions
            target_H, target_W = input_H, input_W

        with torch.no_grad():
            
            if self.use_endosynth:
                # EndoSynth only returns depth [B*num_images,1,H,W]
                depth = self.geometry_model.infer(image_flat)  # [B*num_images,1,H,W]
                depth = depth.squeeze(1)  # [B*num_images,H,W]
                
                # Get depth dimensions (may differ from input due to EndoSynth internal resizing)
                depth_H, depth_W = depth.shape[-2:]
                
                # Create dummy normals (forward-looking normals) matching depth dimensions
                normals = torch.zeros(B * num_images, depth_H, depth_W, 3, device=depth.device, dtype=depth.dtype)
                normals[:, :, :, 2] = 1.0  # [0, 0, 1] normal pointing forward
                
                # Create dummy intrinsics based on target dimensions (will be resized to match)
                intrinsics = torch.zeros(B * num_images, 3, 3, device=depth.device, dtype=depth.dtype)
                intrinsics[:, 0, 0] = target_W * 0.7  # fx (reasonable default)
                intrinsics[:, 1, 1] = target_H * 0.7  # fy
                intrinsics[:, 0, 2] = target_W / 2.0  # cx
                intrinsics[:, 1, 2] = target_H / 2.0  # cy
                intrinsics[:, 2, 2] = 1.0  # homogeneous coordinate
            else:
                # MoGe model returns depth, normals, and intrinsics
                mogeout = self.geometry_model.infer(image_flat)  # [B*num_images,3,H,W] or [B,3,H,W]

                # Extract depth [B*num_images,H,W] and normals [B*num_images,H,W,3]
                depth = mogeout["depth"]  # [B*num_images,H,W] or [B,H,W]
                normals = mogeout["normal"]  # [B*num_images,H,W,3] or [B,H,W,3]
                intrinsics = mogeout["intrinsics"].clone()  # [B*num_images,3,3] or [B,3,3]
                
                # Update intrinsics based on target dimensions
                intrinsics[:, 0, 2] = target_W / 2.0  # cx (width/2)
                intrinsics[:, 1, 2] = target_H / 2.0  # cy (height/2)
                intrinsics[:, :2, :2] = intrinsics[:, :2, :2] * 500
        
        # Sanitize potential NaN/Inf from the geometry model
        depth = torch.nan_to_num(depth, nan=0.0, posinf=1e6, neginf=-1e6)
        normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
        intrinsics = torch.nan_to_num(intrinsics, nan=0.0, posinf=1e6, neginf=-1e6)

        # Resize to target dimensions if needed (always resize to match input or specified target)
        if depth.shape[-2:] != (target_H, target_W):
            resizer = transforms.Resize((target_H, target_W))
            depth = resizer(depth.unsqueeze(1)).squeeze(1)  # [B*num_images,H,W] or [B,H,W]
            # Resize normals - reshape for resize then back
            normals = normals.permute(0, 3, 1, 2)  # [B*num_images,3,H,W] or [B,3,H,W]
            normals = resizer(normals)  # [B*num_images,3,H,W] or [B,3,H,W]
            normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
            normals = F.normalize(
                normals, p=2, dim=1, eps=1e-6
            )  # Re-normalize after resize
        else:
            normals = normals.permute(0, 3, 1, 2)  # [B*num_images,3,H,W] or [B,3,H,W]
            normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
            normals = F.normalize(normals, p=2, dim=1, eps=1e-6)

        # Add channel dimension to depth
        depth = depth.unsqueeze(1)  # [B*num_images,1,H,W] or [B,1,H,W]

        # Reshape results if input was a pair
        if is_pair:
            # Reshape from [B*2, ...] to [B, 2, ...]
            # depth: [B*num_images, 1, H, W] -> [B, num_images, 1, H, W]
            # normals: [B*num_images, 3, H, W] -> [B, num_images, 3, H, W]
            # intrinsics: [B*num_images, 3, 3] -> [B, num_images, 3, 3]
            depth = depth.view(B, num_images, *depth.shape[1:])  # [B,2,1,H,W]
            normals = normals.view(B, num_images, *normals.shape[1:])  # [B,2,3,H,W]
            intrinsics = intrinsics.view(B, num_images, *intrinsics.shape[1:])  # [B,2,3,3]

        do_norm = (
            self.return_normalized_depth
            if return_normalized is None
            else bool(return_normalized)
        )
        if do_norm:
            min_val = depth.min()
            max_val = depth.max()
            if max_val > min_val:
                depth = (depth - min_val) / (max_val - min_val)
            else:
                depth = torch.zeros_like(depth)

        return depth, -normals, intrinsics
