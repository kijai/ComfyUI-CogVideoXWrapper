# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import cv2
import torch
import flow_vis

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
#from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames)


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 1,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        self.vtxt_path = os.path.join(save_dir, "videos.txt")
        self.ttxt_path = os.path.join(save_dir, "trackings.txt")
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame: int = 0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        rigid_part = None,
        video_depth = None # (B,T,C,H,W)
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )

        if video_depth is not None:
            video_depth = (video_depth*255).cpu().numpy().astype(np.uint8)
            video_depth = ([cv2.applyColorMap(video_depth[0,i,0], cv2.COLORMAP_INFERNO) 
                            for i in range(video_depth.shape[1])])
            video_depth = np.stack(video_depth, axis=0)
            video_depth = torch.from_numpy(video_depth).permute(0, 3, 1, 2)[None]

        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        tracking_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            rigid_part=rigid_part
        )

        if save_video:
            # import ipdb; ipdb.set_trace()
            tracking_dir = os.path.join(self.save_dir, "tracking")
            if not os.path.exists(tracking_dir):
                os.makedirs(tracking_dir)
            self.save_video(tracking_video, filename=filename+"_tracking", 
                            savedir=tracking_dir, writer=writer, step=step)
            # with open(self.ttxt_path, 'a') as file:
            #     file.write(f"tracking/{filename}_tracking.mp4\n")

            videos_dir = os.path.join(self.save_dir, "videos")
            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)
            self.save_video(video, filename=filename, 
                            savedir=videos_dir, writer=writer, step=step)
            # with open(self.vtxt_path, 'a') as file:
            #     file.write(f"videos/{filename}.mp4\n")
            if video_depth is not None:
                self.save_video(video_depth, filename=filename+"_depth", 
                                savedir=os.path.join(self.save_dir, "depth"), writer=writer, step=step)
        return tracking_video

    def save_video(self, video, filename, savedir=None, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                f"{filename}",
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            # clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)
            clip = ImageSequenceClip(wide_list, fps=self.fps)

            # Write the video file
            if savedir is None:
                save_path = os.path.join(self.save_dir, f"{filename}.mp4")
            else:
                save_path = os.path.join(savedir, f"{filename}.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame: int = 0,
        compensate_for_camera_motion=False,
        rigid_part=None,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 3
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        # for rgb in video:
        #     res_video.append(rgb.copy())
        
        # create a blank tensor with the same shape as the video
        for rgb in video:
            black_frame = np.zeros_like(rgb.copy(), dtype=rgb.dtype)
            res_video.append(black_frame)

        vector_colors = np.zeros((T, N, 3))

        if self.mode == "optical_flow":

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])

        elif segm_mask is None:
            if self.mode == "rainbow":
                x_min, x_max = tracks[0, :, 0].min(), tracks[0, :, 0].max()
                y_min, y_max = tracks[0, :, 1].min(), tracks[0, :, 1].max()

                z_inv = 1/tracks[0, :, 2]
                z_min, z_max = np.percentile(z_inv, [2, 98])
                
                norm_x = plt.Normalize(x_min, x_max)
                norm_y = plt.Normalize(y_min, y_max)
                norm_z = plt.Normalize(z_min, z_max)

                for n in range(N):
                    r = norm_x(tracks[0, n, 0])
                    g = norm_y(tracks[0, n, 1])
                    # r = 0
                    # g = 0
                    b = norm_z(1/tracks[0, n, 2])
                    color = np.array([r, g, b])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                x_min, x_max = tracks[0, :, 0].min(), tracks[0, :, 0].max()
                y_min, y_max = tracks[0, :, 1].min(), tracks[0, :, 1].max()
                z_min, z_max = tracks[0, :, 2].min(), tracks[0, :, 2].max()

                norm_x = plt.Normalize(x_min, x_max)
                norm_y = plt.Normalize(y_min, y_max)
                norm_z = plt.Normalize(z_min, z_max)

                for n in range(N):
                    r = norm_x(tracks[0, n, 0])
                    g = norm_y(tracks[0, n, 1])
                    b = norm_z(tracks[0, n, 2])
                    color = np.array([r, g, b])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        # Draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        if rigid_part is not None:
            cls_label = torch.unique(rigid_part)
            cls_num = len(torch.unique(rigid_part))
            # visualize the clustering results 
            cmap = plt.get_cmap('jet')  # get the color mapping
            colors = cmap(np.linspace(0, 1, cls_num))  
            colors = (colors[:, :3] * 255) 
            color_map = {lable.item(): color for lable, color in zip(cls_label, colors)}

        # Draw points
        for t in tqdm(range(T)):
            # Create a list to store information for each point
            points_info = []
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                depth = tracks[t, i, 2]  # assume the third dimension is depth
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        points_info.append((i, coord, depth, visibile))
            
            # Sort points by depth, points with smaller depth (closer) will be drawn later
            points_info.sort(key=lambda x: x[2], reverse=True)
            
            for i, coord, _, visibile in points_info:
                if rigid_part is not None:
                    color = color_map[rigid_part.squeeze()[i].item()]
                    cv2.circle(
                        res_video[t],
                        coord,
                        int(self.linewidth * 2),
                        color.tolist(),
                        thickness=-1 if visibile else 2
                        -1,
                    )
                else:
                    # Determine rectangle width based on the distance between adjacent tracks in the first frame
                    if t == 0:
                        distances = np.linalg.norm(tracks[0] - tracks[0, i], axis=1)
                        distances = distances[distances > 0]
                        rect_size = int(np.min(distances))/2
                    
                    # Define coordinates for top-left and bottom-right corners of the rectangle
                    top_left = (int(coord[0] - rect_size), int(coord[1] - rect_size/1.5)) # Rectangle width is 1.5x (video aspect ratio is 1.5:1)
                    bottom_right = (int(coord[0] + rect_size), int(coord[1] + rect_size/1.5))

                    # Draw rectangle
                    cv2.rectangle(
                        res_video[t],
                        top_left,
                        bottom_right,
                        vector_colors[t, i].tolist(),
                        thickness=-1 if visibile else 0
                        -1,
                    )

        # Construct the final rgb sequence
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].tolist(),
                        self.linewidth,
                        cv2.LINE_AA,
                    )
            if self.tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211.0, 0.0, 0.0))

        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
        return rgb
