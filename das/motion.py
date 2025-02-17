import torch
import numpy as np
import math

class CameraMotionGenerator:
    def __init__(self, motion_type, frame_num=49, H=480, W=720, fx=None, fy=None, fov=55, device='cuda'):
        self.motion_type = motion_type
        self.frame_num = frame_num
        self.fov = fov
        self.device = device
        self.W = W
        self.H = H
        self.intr = torch.tensor([
            [0, 0, W / 2],
            [0, 0, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        # if fx, fy not provided
        if not fx or not fy:
            fov_rad = math.radians(fov)
            fx = fy = (W / 2) / math.tan(fov_rad / 2)
 
        self.intr[0, 0] = fx
        self.intr[1, 1] = fy        

    def _apply_poses(self, pts, poses):
        """
        Args:
            pts (torch.Tensor): pointclouds coordinates [T, N, 3]
            intr (torch.Tensor): camera intrinsics [T, 3, 3]
            poses (numpy.ndarray): camera poses [T, 4, 4]
        """
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        intr = self.intr.unsqueeze(0).repeat(self.frame_num, 1, 1).to(torch.float)
        T, N, _ = pts.shape
        ones = torch.ones(T, N, 1, device=self.device, dtype=torch.float)
        pts_hom = torch.cat([pts[:, :, :2], ones], dim=-1)  # (T, N, 3)
        pts_cam = torch.bmm(pts_hom, torch.linalg.inv(intr).transpose(1, 2))  # (T, N, 3)
        pts_cam[:,:, :3] *= pts[:, :, 2:3]

        # to homogeneous
        pts_cam = torch.cat([pts_cam, ones], dim=-1)  # (T, N, 4)
        
        if poses.shape[0] == 1:
            poses = poses.repeat(T, 1, 1)
        elif poses.shape[0] != T:
            raise ValueError(f"Poses length ({poses.shape[0]}) must match sequence length ({T})")
        
        poses = poses.to(torch.float).to(self.device)
        pts_world = torch.bmm(pts_cam, poses.transpose(1, 2))[:, :, :3]  # (T, N, 3)
        pts_proj = torch.bmm(pts_world, intr.transpose(1, 2))  # (T, N, 3)
        pts_proj[:, :, :2] /= pts_proj[:, :, 2:3]

        return pts_proj
    
    def w2s(self, pts, poses):
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        assert poses.shape[0] == self.frame_num
        poses = poses.to(torch.float32).to(self.device)
        T, N, _ = pts.shape  # (T, N, 3)
        intr = self.intr.unsqueeze(0).repeat(self.frame_num, 1, 1)
        # Step 1: 扩展点的维度，使其变成 (T, N, 4)，最后一维填充1 (齐次坐标)
        ones = torch.ones((T, N, 1), device=self.device, dtype=pts.dtype)
        points_world_h = torch.cat([pts, ones], dim=-1)
        points_camera_h = torch.bmm(poses, points_world_h.permute(0, 2, 1))
        points_camera = points_camera_h[:, :3, :].permute(0, 2, 1)

        points_image_h = torch.bmm(points_camera, intr.permute(0, 2, 1))

        uv = points_image_h[:, :, :2] / points_image_h[:, :, 2:3]

        # Step 5: 提取深度 (Z) 并拼接
        depth = points_camera[:, :, 2:3]  # (T, N, 1)
        uvd = torch.cat([uv, depth], dim=-1)  # (T, N, 3)

        return uvd  # 屏幕坐标 + 深度 (T, N, 3)

    def apply_motion_on_pts(self, pts, camera_motion):
        tracking_pts = self._apply_poses(pts.squeeze(), camera_motion).unsqueeze(0)
        return tracking_pts
    
    def set_intr(self, K):
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        self.intr = K.to(self.device)

    def rot_poses(self, angle, axis='y'):
        """
        pts (torch.Tensor): [T, N, 3]
        angle (int): angle of rotation (degree)
        """
        angle_rad = math.radians(angle)
        angles = torch.linspace(0, angle_rad, self.frame_num)
        rot_mats = torch.zeros(self.frame_num, 4, 4)
    
        for i, theta in enumerate(angles):
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            if axis == 'x':
                rot_mats[i] = torch.tensor([
                [1, 0, 0, 0],
                [0, cos_theta, -sin_theta, 0],
                [0, sin_theta, cos_theta, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
            elif axis == 'y':
                rot_mats[i] = torch.tensor([
                    [cos_theta, 0, sin_theta, 0],
                    [0, 1, 0, 0],
                    [-sin_theta, 0, cos_theta, 0],
                    [0, 0, 0, 1]
                ], dtype=torch.float32)
            
            elif axis == 'z':
                rot_mats[i] = torch.tensor([
                    [cos_theta, -sin_theta, 0, 0],
                    [sin_theta, cos_theta, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=torch.float32)
            else:
                raise ValueError("Invalid axis value. Choose 'x', 'y', or 'z'.")
            
        return rot_mats.to(self.device)

    def trans_poses(self, dx, dy, dz):
        """
        params:
        - dx: float, displacement along x axis。
        - dy: float, displacement along y axis。
        - dz: float, displacement along z axis。

        ret:
        - matrices: torch.Tensor
        """
        trans_mats = torch.eye(4).unsqueeze(0).repeat(self.frame_num, 1, 1)  # (n, 4, 4)

        delta_x = dx / (self.frame_num - 1)
        delta_y = dy / (self.frame_num - 1)
        delta_z = dz / (self.frame_num - 1)

        for i in range(self.frame_num):
            trans_mats[i, 0, 3] = i * delta_x
            trans_mats[i, 1, 3] = i * delta_y
            trans_mats[i, 2, 3] = i * delta_z

        return trans_mats.to(self.device)
    

    def _look_at(self, camera_position, target_position):
        # look at direction
        # import ipdb;ipdb.set_trace()
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        # calculate rotation matrix
        up = np.array([0, 1, 0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.vstack([right, up, direction])
        rotation_matrix = np.linalg.inv(rotation_matrix)
        return rotation_matrix

    def spiral_poses(self, radius, forward_ratio = 0.5, backward_ratio = 0.5, rotation_times = 0.1, look_at_times = 0.5):
        """Generate spiral camera poses
        
        Args:
            radius (float): Base radius of the spiral
            forward_ratio (float): Scale factor for forward motion
            backward_ratio (float): Scale factor for backward motion
            rotation_times (float): Number of rotations to complete
            look_at_times (float): Scale factor for look-at point distance
            
        Returns:
            torch.Tensor: Camera poses of shape [num_frames, 4, 4]
        """
        # Generate spiral trajectory
        t = np.linspace(0, 1, self.frame_num)
        r = np.sin(np.pi * t) * radius * rotation_times
        theta = 2 * np.pi * t
        
        # Calculate camera positions
        # Limit y motion for better floor/sky view
        y = r * np.cos(theta) * 0.3  
        x = r * np.sin(theta)
        z = -r
        z[z < 0] *= forward_ratio
        z[z > 0] *= backward_ratio
        
        # Set look-at target
        target_pos = np.array([0, 0, radius * look_at_times])
        cam_pos = np.vstack([x, y, z]).T
        cam_poses = []
        
        for pos in cam_pos:
            rot_mat = self._look_at(pos, target_pos)
            trans_mat = np.eye(4)
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3, 3] = pos
            cam_poses.append(trans_mat[None])
            
        camera_poses = np.concatenate(cam_poses, axis=0)
        return torch.from_numpy(camera_poses).to(self.device)

    def rot(self, pts, angle, axis):
        """
        pts: torch.Tensor, (T, N, 2)
        """
        rot_mats = self.rot_poses(angle, axis)
        pts = self.apply_motion_on_pts(pts, rot_mats)
        return pts
    
    def trans(self, pts, dx, dy, dz):
        if pts.shape[-1] != 3:
            raise ValueError("points should be in the 3d coordinate.")
        trans_mats = self.trans_poses(dx, dy, dz)
        pts = self.apply_motion_on_pts(pts, trans_mats)
        return pts

    def spiral(self, pts, radius):
        spiral_poses = self.spiral_poses(radius)
        pts = self.apply_motion_on_pts(pts, spiral_poses)
        return pts

    def get_default_motion(self):
        if self.motion_type == 'none':
            motion = torch.eye(4).unsqueeze(0).repeat(self.frame_num, 1, 1).to(self.device)
        elif self.motion_type == 'trans':
            motion = self.trans_poses(0.02, 0, 0)
        elif self.motion_type == 'spiral':
            motion = self.spiral_poses(1)
        elif self.motion_type == 'rot':
            motion = self.rot_poses(-25, 'y')
        else:
            raise ValueError(f'camera_motion must be in [trans, spiral, rot], but get {self.motion_type}.')
    
        return motion

class ObjectMotionGenerator:
    def __init__(self, num_frames=49, device="cuda:0"):
        """Initialize ObjectMotionGenerator
        
        Args:
            device (str): Device to run on
        """
        self.device = device
        self.num_frames = num_frames
        
    def _get_points_in_mask(self, pred_tracks, mask):
        """Get points that fall within the mask in first frame
        
        Args:
            pred_tracks (torch.Tensor): [num_frames, num_points, 3]
            mask (torch.Tensor): [H, W] binary mask
            
        Returns:
            torch.Tensor: Boolean mask of selected points [num_points]
        """
        first_frame_points = pred_tracks[0]  # [num_points, 3]
        xy_points = first_frame_points[:, :2]  # [num_points, 2]
        
        # Convert xy coordinates to pixel indices
        xy_pixels = xy_points.round().long()  # Convert to integer pixel coordinates
        
        # Clamp coordinates to valid range
        xy_pixels[:, 0].clamp_(0, mask.shape[1] - 1)  # x coordinates
        xy_pixels[:, 1].clamp_(0, mask.shape[0] - 1)  # y coordinates
        
        # Get mask values at point locations
        points_in_mask = mask[xy_pixels[:, 1], xy_pixels[:, 0]]  # Index using y, x order
        
        return points_in_mask
        
    def generate_motion(self, mask, motion_type, distance, num_frames=49):
        """Generate motion dictionary for the given parameters
        
        Args:
            mask (torch.Tensor): [H, W] binary mask
            motion_type (str): Motion direction ('up', 'down', 'left', 'right')
            distance (float): Total distance to move
            num_frames (int): Number of frames
            
        Returns:
            dict: Motion dictionary containing:
                - mask (torch.Tensor): Binary mask
                - motions (torch.Tensor): Per-frame motion vectors [num_frames, 4, 4]
        """

        self.num_frames = num_frames
        # Define motion template vectors
        template = {
            "none": torch.tensor([0, 0, 0]),
            'up': torch.tensor([0, -1, 0]),
            'down': torch.tensor([0, 1, 0]),
            'left': torch.tensor([-1, 0, 0]),
            'right': torch.tensor([1, 0, 0]),
            'front': torch.tensor([0, 0, 1]),
            'back': torch.tensor([0, 0, -1])
        }
        
        if motion_type not in template:
            raise ValueError(f"Unknown motion type: {motion_type}")
            
        # Move mask to device
        mask = mask.to(self.device)
        
        # Generate per-frame motion matrices
        motions = []
        base_vec = template[motion_type].to(self.device) * distance
        
        for frame_idx in range(num_frames):
            # Calculate interpolation factor (0 to 1)
            t = frame_idx / (num_frames - 1)
            
            # Create motion matrix for current frame
            current_motion = torch.eye(4, device=self.device)
            current_motion[:3, 3] = base_vec * t
            motions.append(current_motion)
            
        motions = torch.stack(motions)  # [num_frames, 4, 4]
        
        return {
            'mask': mask,
            'motions': motions
        }
        
    def apply_motion(self, pred_tracks, motion_dict, tracking_method="spatracker"):
        """Apply motion to selected points
        
        Args:
            pred_tracks (torch.Tensor): [num_frames, num_points, 3] for spatracker
                                      or [T, H, W, 3] for moge
            motion_dict (dict): Motion dictionary containing mask and motions
            tracking_method (str): "spatracker" or "moge"
            
        Returns:
            torch.Tensor: Modified pred_tracks with same shape as input
        """
        pred_tracks = pred_tracks.to(self.device).float()
        
        if tracking_method == "moge":

            H = pred_tracks.shape[0]
            W = pred_tracks.shape[1]
            
            initial_points = pred_tracks  # [H, W, 3]
            selected_mask = motion_dict['mask']
            valid_selected = ~torch.any(torch.isnan(initial_points), dim=2) & selected_mask
            valid_selected = valid_selected.reshape([-1])
            modified_tracks = pred_tracks.clone().reshape(-1, 3).unsqueeze(0).repeat(self.num_frames, 1, 1)
            # import ipdb;ipdb.set_trace()
            for frame_idx in range(self.num_frames):
                # Get current frame motion
                motion_mat = motion_dict['motions'][frame_idx]
                # Moge's pointcloud is scale-invairant
                motion_mat[0, 3] /= W
                motion_mat[1, 3] /= H
                # Apply motion to selected points
                points = modified_tracks[frame_idx, valid_selected]
                # Convert to homogeneous coordinates
                points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                # Apply transformation
                transformed_points = torch.matmul(points_homo, motion_mat.T)
                # Convert back to 3D coordinates
                modified_tracks[frame_idx, valid_selected] = transformed_points[:, :3]
            return modified_tracks
            
        else:
            points_in_mask = self._get_points_in_mask(pred_tracks, motion_dict['mask'])
            modified_tracks = pred_tracks.clone()
            
            for frame_idx in range(pred_tracks.shape[0]):
                motion_mat = motion_dict['motions'][frame_idx]
                points = modified_tracks[frame_idx, points_in_mask]
                points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                transformed_points = torch.matmul(points_homo, motion_mat.T)
                modified_tracks[frame_idx, points_in_mask] = transformed_points[:, :3]
            
            return modified_tracks