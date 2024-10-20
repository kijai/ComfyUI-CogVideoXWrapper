import numpy as np
import cv2
import torch

# Note that the coordinates passed to the model must not exceed 256.
# xy range 256
PROVIDED_TRAJS = {
    "circle1": [
        [120, 194],
        [144, 193],
        [155, 189],
        [158, 170],
        [160, 153],
        [159, 123],
        [152, 113],
        [136, 100],
        [124, 100],
        [108, 100],
        [101, 106],
        [90, 110],
        [84, 129],
        [79, 146],
        [78, 165],
        [83, 182],
        [87, 189],
        [94, 192],
        [100, 194],
        [106, 194],
        [112, 194],
        [118, 195],
    ],
    "circle2": [
        [100, 127],
        [105, 117],
        [122, 117],
        [132, 129],
        [133, 158],
        [125, 181],
        [108, 189],
        [92, 185],
        [84, 179],
        [79, 163],
        [75, 142],
        [73, 118],
        [75, 82],
        [91, 63],
        [115, 52],
        [139, 46],
        [154, 55],
        [167, 93],
        [175, 112],
        [177, 137],
        [177, 158],
        [177, 171],
        [175, 188],
        [173, 204],
    ],
    "coaster": [
        [40, 208],
        [40, 148],
        [40, 100],
        [52, 58],
        [60, 57],
        [74, 68],
        [78, 90],
        [84, 123],
        [88, 148],
        [96, 168],
        [100, 181],
        [102, 188],
        [105, 192],
        [113, 118],
        [119, 80],
        [128, 68],
        [145, 109],
        [149, 155],
        [157, 175],
        [161, 184],
        [164, 184],
        [172, 166],
        [183, 107],
        [189, 84],
        [198, 76],
    ],
    "dance": [
        [81, 112],
        [86, 112],
        [92, 112],
        [100, 113],
        [102, 114],
        [97, 115],
        [92, 114],
        [86, 112],
        [81, 112],
        [80, 112],
        [84, 113],
        [89, 114],
        [95, 114],
        [101, 114],
        [102, 114],
        [103, 124],
        [105, 137],
        [109, 156],
        [114, 172],
        [119, 180],
        [124, 184],
        [131, 181],
        [140, 168],
        [146, 152],
        [150, 128],
        [151, 117],
        [152, 116],
        [156, 116],
        [163, 115],
        [169, 116],
        [175, 116],
        [173, 116],
        [167, 116],
        [162, 114],
        [157, 114],
        [152, 115],
        [156, 115],
        [163, 115],
        [168, 115],
        [174, 116],
        [175, 116],
        [168, 116],
        [162, 116],
        [152, 114],
        [149, 134],
        [145, 156],
        [139, 168],
        [130, 183],
        [118, 180],
        [112, 170],
        [107, 151],
        [102, 128],
        [103, 117],
        [96, 113],
        [88, 113],
        [83, 112],
        [80, 112],
    ],
    "infinity": [
        [60, 141],
        [71, 127],
        [92, 120],
        [112, 123],
        [130, 145],
        [145, 163],
        [167, 178],
        [189, 187],
        [206, 176],
        [213, 147],
        [208, 124],
        [190, 112],
        [176, 111],
        [158, 124],
        [145, 147],
        [125, 172],
        [104, 189],
        [72, 189],
        [59, 184],
        [55, 153],
        [57, 140],
        [75, 119],
        [112, 118],
        [129, 142],
        [149, 163],
        [168, 180],
        [194, 186],
        [206, 175],
        [211, 159],
        [212, 149],
        [212, 134],
        [206, 122],
        [180, 112],
        [163, 116],
        [149, 138],
        [128, 170],
        [108, 184],
        [86, 190],
        [63, 181],
        [57, 152],
        [57, 139],
    ],
    "pause": [
        [98, 186],
        [100, 188],
        [98, 186],
        [100, 188],
        [101, 187],
        [104, 187],
        [111, 184],
        [116, 176],
        [125, 162],
        [132, 140],
        [136, 119],
        [137, 104],
        [138, 96],
        [139, 94],
        [140, 94],
        [140, 96],
        [138, 98],
        [138, 96],
        [136, 94],
        [137, 92],
        [140, 92],
        [144, 92],
        [149, 92],
        [152, 92],
        [151, 92],
        [147, 92],
        [142, 92],
        [140, 92],
        [139, 95],
        [139, 105],
        [141, 122],
        [142, 143],
        [140, 167],
        [136, 184],
        [135, 188],
        [132, 195],
        [132, 192],
        [131, 192],
        [131, 192],
        [130, 192],
        [130, 195],
    ],
    "shake": [
        [103, 89],
        [104, 89],
        [106, 89],
        [107, 89],
        [108, 89],
        [109, 89],
        [110, 89],
        [111, 89],
        [112, 89],
        [113, 89],
        [114, 89],
        [115, 89],
        [116, 89],
        [117, 89],
        [118, 89],
        [119, 89],
        [120, 89],
        [122, 89],
        [123, 89],
        [124, 89],
        [125, 89],
        [126, 89],
        [127, 88],
        [128, 88],
        [129, 88],
        [130, 88],
        [131, 88],
        [133, 87],
        [136, 86],
        [137, 86],
        [138, 86],
        [139, 86],
        [140, 86],
        [141, 86],
        [142, 86],
        [143, 86],
        [144, 86],
        [145, 86],
        [146, 87],
        [147, 87],
        [148, 87],
        [149, 87],
        [148, 87],
        [146, 87],
        [145, 88],
        [144, 88],
        [142, 89],
        [141, 89],
        [140, 90],
        [140, 91],
        [138, 91],
        [137, 92],
        [136, 92],
        [136, 93],
        [135, 93],
        [134, 93],
        [133, 93],
        [132, 93],
        [131, 93],
        [130, 93],
        [129, 93],
        [128, 93],
        [127, 92],
        [125, 92],
        [124, 92],
        [123, 92],
        [122, 92],
        [121, 92],
        [120, 92],
        [119, 92],
        [118, 92],
        [117, 92],
        [116, 92],
        [115, 92],
        [113, 92],
        [112, 92],
        [111, 92],
        [110, 92],
        [109, 92],
        [108, 92],
        [108, 91],
        [108, 90],
        [109, 90],
        [110, 90],
        [111, 89],
        [112, 89],
        [113, 89],
        [114, 89],
        [115, 89],
        [115, 88],
        [116, 88],
        [117, 88],
        [118, 88],
        [118, 87],
        [119, 87],
        [120, 87],
        [121, 87],
        [122, 86],
        [123, 86],
        [124, 86],
        [125, 86],
        [126, 85],
        [127, 85],
        [128, 85],
        [129, 85],
        [130, 85],
        [131, 85],
        [132, 85],
        [133, 85],
        [134, 85],
        [135, 85],
        [136, 85],
        [137, 85],
        [138, 85],
        [139, 85],
        [140, 85],
        [141, 85],
        [142, 85],
        [143, 85],
        [143, 84],
        [144, 84],
        [145, 84],
        [146, 84],
        [147, 84],
        [148, 84],
        [149, 84],
        [148, 84],
        [147, 84],
        [145, 84],
        [144, 84],
        [143, 84],
        [142, 84],
        [141, 84],
        [140, 85],
        [139, 85],
        [138, 85],
        [137, 86],
        [136, 86],
        [136, 87],
        [135, 87],
        [134, 87],
        [133, 87],
        [132, 88],
        [131, 88],
        [130, 88],
        [129, 88],
        [129, 89],
        [128, 89],
        [127, 89],
        [126, 89],
        [125, 89],
        [124, 90],
        [123, 90],
        [122, 90],
        [121, 90],
        [120, 91],
        [119, 91],
        [118, 91],
        [117, 91],
        [116, 91],
        [115, 91],
        [114, 91],
        [113, 91],
        [112, 91],
        [111, 91],
        [110, 91],
        [109, 91],
        [109, 90],
        [108, 90],
        [110, 90],
        [111, 90],
        [113, 90],
        [114, 90],
        [115, 90],
        [116, 90],
        [118, 90],
        [120, 90],
        [121, 90],
        [122, 90],
        [123, 90],
        [124, 90],
        [126, 90],
        [127, 90],
        [128, 90],
        [129, 90],
        [130, 90],
        [131, 90],
        [132, 90],
        [133, 90],
        [134, 90],
        [135, 90],
        [136, 90],
        [137, 90],
        [138, 90],
        [139, 90],
        [140, 90],
        [141, 89],
        [142, 89],
        [143, 89],
        [144, 89],
        [145, 89],
        [146, 89],
        [147, 89],
        [147, 89],
        [147, 89],
    ],
    "spiral": [
        [16, 152],
        [23, 138],
        [39, 122],
        [54, 115],
        [75, 118],
        [88, 130],
        [93, 150],
        [89, 176],
        [75, 184],
        [63, 177],
        [65, 152],
        [77, 135],
        [98, 121],
        [116, 120],
        [135, 127],
        [148, 136],
        [156, 145],
        [160, 165],
        [158, 176],
        [138, 187],
        [133, 185],
        [129, 148],
        [140, 133],
        [156, 120],
        [177, 118],
        [197, 118],
        [214, 119],
        [225, 118],
    ],
}

def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.
    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.
    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.
    Args:
        kernel_size (int):
    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack(
        (
            xx.reshape((kernel_size * kernel_size, 1)),
            yy.reshape(kernel_size * kernel_size, 1),
        )
    ).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).
    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
    Returns:
        ndarray: Rotated sigma matrix.
    """
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel

size = 99
sigma = 10
blur_kernel = bivariate_Gaussian(size, sigma, sigma, 0, grid=None, isotropic=True)
blur_kernel = blur_kernel / blur_kernel[size // 2, size // 2]

canvas_width, canvas_height = 256, 256

def get_flow(points, optical_flow, video_len):
    for i in range(video_len - 1):
        p = points[i]
        p1 = points[i + 1]
        optical_flow[i + 1, p[1], p[0], 0] = p1[0] - p[0]
        optical_flow[i + 1, p[1], p[0], 1] = p1[1] - p[1]

    return optical_flow


def process_points(points, frames=49):
    defualt_points = [[128, 128]] * frames

    if len(points) < 2:
        return defualt_points

    elif len(points) >= frames:
        skip = len(points) // frames
        return points[::skip][: frames - 1] + points[-1:]
    else:
        insert_num = frames - len(points)
        insert_num_dict = {}
        interval = len(points) - 1
        n = insert_num // interval
        m = insert_num % interval
        for i in range(interval):
            insert_num_dict[i] = n
        for i in range(m):
            insert_num_dict[i] += 1

        res = []
        for i in range(interval):
            insert_points = []
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            delta_x = x1 - x0
            delta_y = y1 - y0
            for j in range(insert_num_dict[i]):
                x = x0 + (j + 1) / (insert_num_dict[i] + 1) * delta_x
                y = y0 + (j + 1) / (insert_num_dict[i] + 1) * delta_y
                insert_points.append([int(x), int(y)])

            res += points[i : i + 1] + insert_points
        res += points[-1:]
        return res


def read_points_from_list(traj_list, video_len=16, reverse=False):
    points = []
    for point in traj_list:
        if isinstance(point, str):
            x, y = point.strip().split(",")
        else:
            x, y = point[0], point[1]
        points.append((int(x), int(y)))
    if reverse:
        points = points[::-1]

    if len(points) > video_len:
        skip = len(points) // video_len
        points = points[::skip]
    points = points[:video_len]

    return points


def read_points_from_file(file, video_len=16, reverse=False):
    with open(file, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        x, y = line.strip().split(",")
        points.append((int(x), int(y)))
    if reverse:
        points = points[::-1]

    if len(points) > video_len:
        skip = len(points) // video_len
        points = points[::skip]
    points = points[:video_len]

    return points


def process_traj(trajs_list, num_frames, video_size, device="cpu"):
    if trajs_list and trajs_list[0] and (not isinstance(trajs_list[0][0], (list, tuple))):
        tmp = trajs_list
        trajs_list = [tmp]

    optical_flow = np.zeros((num_frames, video_size[0], video_size[1], 2), dtype=np.float32)
    processed_points = []
    for traj_list in trajs_list:
        points = read_points_from_list(traj_list, video_len=num_frames)
        xy_range = 256
        h, w = video_size
        points = process_points(points, num_frames)
        points = [[int(w * x / xy_range), int(h * y / xy_range)] for x, y in points]
        optical_flow = get_flow(points, optical_flow, video_len=num_frames)
        processed_points.append(points)

    print(f"received {len(trajs_list)} trajectorie(s)")

    for i in range(1, num_frames):
        optical_flow[i] = cv2.filter2D(optical_flow[i], -1, blur_kernel)

    optical_flow = torch.tensor(optical_flow).to(device)

    return optical_flow, processed_points


def add_provided_traj(traj_name):
    global traj_list
    traj_list = PROVIDED_TRAJS[traj_name]
    traj_str = [f"{traj}" for traj in traj_list]
    return ", ".join(traj_str)


def scale_traj_list_to_256(traj_list, canvas_width, canvas_height):
    scale_x = 256 / canvas_width
    scale_y = 256 / canvas_height
    scaled_traj_list = [[int(x * scale_x), int(y * scale_y)] for x, y in traj_list]
    return scaled_traj_list