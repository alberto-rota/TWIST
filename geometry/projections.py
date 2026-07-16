import torch
import torch.nn as nn
import torch.nn.functional as F
import geometry.transforms as geometry


class BackProject(nn.Module):
    """Back-projects 2D points to 3D space using depth information.

    Args:
        height (int): Image height
        width (int): Image width
    """

    def __init__(self, height: int, width: int):
        super(BackProject, self).__init__()

        self.height = height
        self.width = width

        # Create meshgrid of pixel coordinates
        x = torch.arange(0, width).float()
        y = torch.arange(0, height).float()
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        # Register buffer for pixel coordinates [3, H*W]
        self.register_buffer(
            "pix_coords",
            torch.stack(
                [xx.reshape(-1), yy.reshape(-1), torch.ones_like(xx).reshape(-1)], dim=0
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        invK: torch.Tensor,
        points_match: torch.Tensor = None,
        batch_idx_match: torch.Tensor = None,
    ) -> dict:
        """
        Back-project image points to 3D space.

        Args:
            image: RGB image [B, 3, H, W]
            depth: Depth map [B, 1, H, W]
            invK: Inverse camera intrinsics [B, 3, 3]
            points_match: Specific pixel coordinates to project in either:
                - [B, N, 2] format (same number of points per batch)
                - [BN, 2] format (variable number of points per batch)
            batch_idx_match: If points_match is [BN, 2], indicates batch indices [BN, 1]

        Returns:
            dict containing:
                - xyz1: Homogeneous 3D points [B, 4, H*W]
                - depth: Flattened depth [B, 1, H*W]
                - rgb: Flattened RGB values [B, 3, H*W]
                - points_match_3d: 3D coordinates of matched points (format matches input)
                - batch_idx_match: Only returned if input used non-batched format
        """
        batch_size = depth.size(0)

        # Expand pixel coordinates to batch dimension [B, 3, H*W]
        pix_coords = self.pix_coords.unsqueeze(0).expand(batch_size, -1, -1)

        # Transform pixels to camera coordinates using batched matrix multiplication
        cam_points_plane = torch.bmm(invK, pix_coords)  # [B, 3, H*W]

        # Scale by depth
        depth_flat = depth.view(batch_size, 1, -1)  # [B, 1, H*W]
        cam_points = cam_points_plane * depth_flat  # [B, 3, H*W]

        # Create homogeneous coordinates
        ones = torch.ones_like(depth_flat)  # [B, 1, H*W]
        cam_points = torch.cat([cam_points, ones], dim=1)  # [B, 4, H*W]

        # Flatten RGB image
        rgb_flat = image.view(batch_size, 3, -1)  # [B, 3, H*W]

        result = {
            "xyz1": cam_points,
            "depth": depth_flat,
            "rgb": rgb_flat,
        }

        # Handle points_match if provided
        if points_match is not None:
            # Check the format of points_match
            if len(points_match.shape) == 3:  # [B, N, 2] format
                # Convert points_match to homogeneous coordinates [B, N, 3]
                points_match_homo = torch.cat(
                    [points_match, torch.ones_like(points_match[..., :1])], dim=-1
                )

                # Transpose for batch matrix multiplication [B, 3, N]
                points_match_homo = points_match_homo.transpose(1, 2)

                # Transform to camera coordinates
                points_match_cam = torch.bmm(invK, points_match_homo)

                # Get depth values at the matched points
                # First get the pixel coordinates as integers
                points_match_px = points_match.int()
                batch_idx = (
                    torch.arange(batch_size, device=points_match.device)
                    .view(-1, 1)
                    .expand(-1, points_match.size(1))
                )

                # Sample depth values at these coordinates
                points_match_depth = (
                    depth[
                        batch_idx.reshape(-1),
                        torch.zeros_like(batch_idx.reshape(-1)),  # channel index
                        points_match_px[..., 1].reshape(-1).clamp(0, self.height - 1),
                        points_match_px[..., 0].reshape(-1).clamp(0, self.width - 1),
                    ]
                    .reshape(batch_size, -1)
                    .unsqueeze(1)
                )

                # Scale by depth
                points_match_3d = points_match_cam * points_match_depth  # [B, 3, N]

                # Add homogeneous coordinate
                ones_match = torch.ones_like(points_match_depth)  # [B, 1, N]
                points_match_3d = torch.cat(
                    [points_match_3d, ones_match], dim=1
                )  # [B, 4, N]

                result["points_match_3d"] = points_match_3d

            elif len(points_match.shape) == 2:  # [BN, 2] format
                assert (
                    batch_idx_match is not None
                ), "batch_idx_match must be provided for [BN, 2] format"
                assert (
                    batch_idx_match.shape[0] == points_match.shape[0]
                ), "batch_idx_match must have same length as points_match"

                # Get batch indices (flattened)s
                batch_indices = batch_idx_match.squeeze(-1).int()

                # Convert to homogeneous coordinates [BN, 3]
                points_match_homo = torch.cat(
                    [points_match, torch.ones_like(points_match[:, :1])], dim=1
                )  # [BN, 3]

                # Get invK for each point based on batch_indices
                point_invK = invK[batch_indices]  # [BN, 3, 3]

                # Transform to camera coordinates using batched matrix multiplication
                points_match_cam = torch.bmm(
                    point_invK, points_match_homo.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # [BN, 3]

                # Clamp coordinates to valid image range
                point_y = points_match[:, 1].long().clamp(0, self.height - 1)
                point_x = points_match[:, 0].long().clamp(0, self.width - 1)

                # Sample depth values at these coordinates
                points_match_depth = depth[
                    batch_indices, 0, point_y, point_x  # channel index
                ].unsqueeze(
                    -1
                )  # [BN, 1]

                # Scale by depth
                points_match_3d = points_match_cam * points_match_depth  # [BN, 3]

                # Add homogeneous coordinate
                ones_match = torch.ones_like(points_match_depth)  # [BN, 1]
                points_match_3d = torch.cat(
                    [points_match_3d, ones_match], dim=1
                )  # [BN, 4]

                result["points_match_3d"] = points_match_3d
                result["batch_idx_match"] = batch_idx_match

        return result


class Project(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.width = width
        self.height = height

    def forward(
        self,
        cloud,
        rgb_vec,
        K,
        T,
        points_match_3d=None,
        batch_idx_match=None,
        missing_value=0,
        median_kernel_size=5,
        return_artifacts=False,
        return_mask=False,
    ):
        """
        Project 3D points to 2D image space.

        Args:
            cloud: 3D point cloud [B, 4, N]
            rgb_vec: RGB values for each point [B, 3, N]
            K: Camera intrinsics [B, 3, 3]
            T: Camera pose [B, 4, 4] or [B, 6] (Euler)
            points_match_3d: 3D points to track in either:
                - [B, 4, N] format (same number of points per batch)
                - [BN, 4] format (variable number of points per batch)
            batch_idx_match: If points_match_3d is [BN, 4], indicates batch indices [BN, 1]
            missing_value: Value to fill in missing pixels
            median_kernel_size: Size of kernel for median filtering
            return_artifacts: Whether to return intermediate artifacts
            return_mask: Whether to return visibility mask

        Returns:
            dict: Dictionary containing warped image, points, and other outputs
        """
        B, _, N = cloud.shape
        device = cloud.device

        if T.shape[1] == 6:
            T = geometry.euler2mat(T)
        T = torch.inverse(T)
        # Project point cloud to camera space
        cloud_cam = torch.bmm(T, cloud)  # B x 4 x N
        proj = torch.bmm(K, cloud_cam[:, :3, :])  # B x 3 x N
        uv = proj[:, :2, :] / proj[:, 2:3, :]  # B x 2 x N
        depth = cloud_cam[:, 2, :]  # B x N

        # Clamp projected coordinates to image boundaries
        v = uv[:, 1, :].int().clamp(0, self.height - 1)
        u = uv[:, 0, :].int().clamp(0, self.width - 1)

        # Compute linear indices for scatter operations
        batch_offset = (torch.arange(B, device=device) * self.height * self.width).view(
            B, 1
        )
        linear_idx = batch_offset + v * self.width + u  # B x N

        # Flatten for scatter operations
        flat_linear_idx = linear_idx.reshape(-1).long()  # (B*N,)
        flat_depth = depth.reshape(-1).long()  # (B*N,)
        flat_rgb = rgb_vec.permute(0, 2, 1).reshape(-1, 3)  # (B*N, 3)

        # Depth buffer initialization for scatter_reduce.
        # NOTE: the original GateTracker code used ``torch.full(..., float("inf")).long()``
        # as the +inf sentinel for the amin z-buffer. Casting float ``inf`` to
        # int64 is undefined behaviour in C++: CUDA saturates to LONG_MAX (so the
        # buffer works), but CPU (x86) wraps to LONG_MIN — which then wins every
        # ``amin(..., include_self=True)`` and rejects all points, rendering an
        # all-holes (black) frame. Use the explicit int64 max so the z-buffer is
        # correct on both backends (verified in workshops/36).
        depth_buffer = torch.full(
            (B * self.height * self.width,),
            torch.iinfo(torch.long).max,
            device=device,
            dtype=torch.long,
        )
        # Use scatter_reduce to find the minimum depth per pixel
        depth_buffer = depth_buffer.scatter_reduce(
            0, flat_linear_idx, flat_depth, reduce="amin", include_self=True
        )

        gathered_depth = depth_buffer[flat_linear_idx]
        # Mask for selecting the closest point per pixel
        mask = torch.isclose(flat_depth, gathered_depth, atol=1e-6)

        # Filter RGB values using the mask
        flat_rgb_filtered = torch.zeros_like(flat_rgb)
        flat_rgb_filtered[mask] = flat_rgb[mask]

        image_flat = -0.001 * torch.ones(
            B * self.height * self.width, 3, device=device, dtype=flat_rgb.dtype
        )
        image_flat = image_flat.index_copy(
            0, flat_linear_idx[mask], flat_rgb_filtered[mask]
        )
        image = image_flat.view(B, self.height, self.width, 3).permute(0, 3, 1, 2)

        # Classification mask initialization:
        # Start with all pixels as holes (0)
        classification_mask = torch.zeros(
            B, self.height, self.width, device=device, dtype=torch.uint8
        )

        # Count the number of points projected to each pixel to identify occlusions
        count_buffer = torch.zeros(
            B * self.height * self.width, device=device, dtype=torch.int32
        )
        ones = torch.ones_like(flat_depth, dtype=torch.int32)
        count_buffer = count_buffer.scatter_reduce(
            0, flat_linear_idx, ones, reduce="sum"
        )
        count_buffer = count_buffer.view(B, self.height, self.width)

        # Populate the classification mask
        classification_mask[count_buffer == 1] = 1  # Valid pixels
        classification_mask[count_buffer > 1] = 2  # Overlapping pixels (occlusions)

        # Expand to match image channels for compatibility with output shape
        classification_mask = classification_mask.unsqueeze(1).expand(-1, 3, -1, -1)

        # Save warped image before hole filling and median filtering if artifacts are requested
        warped_image = image.clone() if return_artifacts else None

        # Inpainting with median filtering integrated
        base_mask = (image[:, :1, :, :] > missing_value).float()  # shape: (B, 1, H, W)
        mask_full = base_mask.expand_as(image)  # shape: (B, C, H, W)
        visibility_confidence = (
            (classification_mask == 1).float()
            + 0.35 * (classification_mask == 2).float()
        )  # [B, 3, H, W]

        img_for_interp = image.clone()
        img_for_interp[mask_full == 0] = 0.0

        _, C, H, W = image.shape
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        inpainted_img = image.clone()
        count_buffer = count_buffer.unsqueeze(1).expand(-1, 3, -1, -1)

        pad = median_kernel_size // 2
        for _ in range(10):
            padded = F.pad(inpainted_img, (pad, pad, pad, pad), mode="reflect")
            patches = padded.unfold(2, median_kernel_size, 1).unfold(
                3, median_kernel_size, 1
            )
            patches = patches.contiguous().view(B, C, H, W, -1)
            median_img, _ = patches.median(dim=-1)
            final_img = inpainted_img.clone()
            final_img[mask_full == 0] = median_img[mask_full == 0]
            inpainted_img = final_img

        # Handle points_match_3d projection if provided
        uv_match = None
        if points_match_3d is not None:
            if len(points_match_3d.shape) == 3:  # [B, 4, N] format (batched)
                _, _, Np = points_match_3d.shape

                # Project points to camera space and then to image
                points_cam = torch.bmm(T, points_match_3d)  # [B, 4, N]
                proj_match = torch.bmm(K, points_cam[:, :3, :])  # [B, 3, N]
                uv_match = proj_match[:, :2, :] / proj_match[:, 2:3, :]  # [B, 2, N]
                uv_match = uv_match.permute(0, 2, 1)  # [B, N, 2]

            elif len(points_match_3d.shape) == 2:  # [BN, 4] format (non-batched)
                assert (
                    batch_idx_match is not None
                ), "batch_idx_match must be provided for [BN, 4] format"

                # Get batch indices
                batch_indices = batch_idx_match.squeeze(-1).int()

                # Get T and K for each point based on batch indices
                point_T = T[batch_indices]  # [BN, 4, 4]
                point_K = K[batch_indices]  # [BN, 3, 3]

                # Project each point to camera space
                points_cam = torch.bmm(point_T, points_match_3d.unsqueeze(-1)).squeeze(
                    -1
                )  # [BN, 4]

                # Project to image plane
                proj_match = torch.bmm(
                    point_K, points_cam[:, :3].unsqueeze(-1)
                ).squeeze(
                    -1
                )  # [BN, 3]

                # Calculate UV coordinates
                z = proj_match[:, 2].clamp(min=1e-10)
                uv_match = proj_match[:, :2] / z.unsqueeze(-1)  # [BN, 2]

        # Prepare the dictionary to return
        output = {}
        expandedmask = (
            (inpainted_img > 0).any(dim=1, keepdim=True).expand(-1, 3, -1, -1).int()
        )

        mask = expandedmask.bool()
        single_channel = mask[:, 0, :, :].float()
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)
        neighbor_count = F.conv2d(single_channel.unsqueeze(1), kernel, padding=1)
        updated_channel = (neighbor_count >= 6).squeeze(1)
        holemask = updated_channel.unsqueeze(1).repeat(1, 3, 1, 1).int()

        if return_mask:
            output["mask"] = holemask
        else:
            output["mask"] = None

        output["raw_mask"] = mask_full.float() if return_artifacts else None
        output["visibility_confidence"] = (
            visibility_confidence if return_artifacts else None
        )
        output["warped"] = inpainted_img * holemask
        # Raw warped image before hole filling and median filtering if artifacts requested
        output["raw"] = warped_image if return_artifacts else None
        output["matches"] = uv_match

        # Include batch_idx_match in output if it was provided
        if batch_idx_match is not None and points_match_3d is not None:
            output["batch_idx_match"] = batch_idx_match

        return output


class Warp(nn.Module):
    """
    Warp module that combines back-projection and forward-projection operations.
    Transforms a source image to a target viewpoint based on depth and camera pose.
    """

    def __init__(self, height, width):
        super(Warp, self).__init__()
        self.height = height
        self.width = width
        self.backproject = BackProject(height, width)
        self.forward_project = Project(height, width)

    def forward(
        self,
        source_image,
        depth_map,
        camera_intrinsics,
        camera_pose,
        return_mask=False,
        return_artifacts=False,
        points_to_match=None,
        batch_idx_match=None,
        median_kernel_size=5,
    ):
        """
        Warp a source image to a target viewpoint based on depth and camera pose.

        Args:
            source_image (torch.Tensor): Source image [B, 3, H, W]
            depth_map (torch.Tensor): Depth map for source image [B, 1, H, W]
            camera_intrinsics (torch.Tensor): Camera intrinsics matrix [B, 3, 3]
            camera_pose (torch.Tensor): Camera pose / transformation [B, 4, 4] or [B, 6] (Euler)
            return_mask (bool): Whether to return visibility mask
            return_artifacts (bool): Whether to return intermediate artifacts
            points_to_match (torch.Tensor, optional): Source points to track in either:
                - [B, N, 2] format (same number of points per batch)
                - [BN, 2] format (variable number of points per batch)
            batch_idx_match (torch.Tensor, optional): If points_to_match is [BN, 2], this tensor [BN, 1]
                indicates which batch each point belongs to
            median_kernel_size (int): Size of kernel for median filtering in forward projection

        Returns:
            dict: Dictionary containing warped image, points, and other outputs
        """
        # Validate input format
        batch_size = source_image.shape[0]

        if points_to_match is not None:
            if len(points_to_match.shape) == 2:  # [BN, 2] format
                assert (
                    points_to_match.shape[1] == 2
                ), "Last dimension of points_to_match must be 2"
                assert (
                    batch_idx_match is not None
                ), "batch_idx_match must be provided when points_to_match has shape [BN, 2]"
                assert (
                    batch_idx_match.shape[0] == points_to_match.shape[0]
                ), "batch_idx_match and points_to_match must have the same first dimension"
                assert (
                    len(batch_idx_match.shape) == 1
                ), "batch_idx_match must have shape [BN,]"
                assert torch.all(batch_idx_match >= 0) and torch.all(
                    batch_idx_match < batch_size
                ), f"batch_idx_match values must be in range [0, {batch_size-1}]"
            elif len(points_to_match.shape) == 3:  # [B, N, 2] format
                assert (
                    points_to_match.shape[0] == batch_size
                ), "Batch size of points_to_match must match source_image"
                assert (
                    points_to_match.shape[2] == 2
                ), "Last dimension of points_to_match must be 2"
                if batch_idx_match is not None:
                    print(
                        "Warning: batch_idx_match is ignored when points_to_match has shape [B, N, 2]"
                    )
                    batch_idx_match = None
            else:
                raise ValueError(
                    f"Invalid shape for points_to_match: {points_to_match.shape}"
                )

        # Back-project source image and/or points to 3D
        backproj_output = self.backproject(
            source_image,
            depth_map,
            torch.inverse(camera_intrinsics),
            points_match=points_to_match,
            batch_idx_match=batch_idx_match,
        )

        # Extract 3D points and RGB values
        cloud = backproj_output["xyz1"]
        rgb_vec = backproj_output["rgb"]

        # Forward-project to create warped image and/or track points
        warp_output = self.forward_project(
            cloud,
            rgb_vec,
            camera_intrinsics,
            camera_pose,
            points_match_3d=backproj_output.get("points_match_3d", None),
            batch_idx_match=backproj_output.get("batch_idx_match", None),
            return_mask=return_mask,
            return_artifacts=return_artifacts,
            median_kernel_size=median_kernel_size,
        )
        warp_output.update(backproj_output)
        return warp_output
