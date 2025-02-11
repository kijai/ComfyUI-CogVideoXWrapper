import torch
from . import basic as utils
import numpy as np
import torchvision.ops as ops
from .basic import print_

def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)

def matmul3(mat1, mat2, mat3):
    return torch.matmul(mat1, torch.matmul(mat2, mat3))

def eye_3x3(B, device='cuda'):
    rt = torch.eye(3, device=torch.device(device)).view(1,3,3).repeat([B, 1, 1])
    return rt

def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4)
    inv = torch.cat([inv, bottom_row], 0)
    return inv

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/(z)+x0
    y = (y*fy)/(z)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy

def apply_pix_T_cam_py(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]

    fx = np.reshape(fx, [B, 1])
    fy = np.reshape(fy, [B, 1])
    x0 = np.reshape(x0, [B, 1])
    y0 = np.reshape(y0, [B, 1])

    EPS = 1e-4
    z = np.clip(z, EPS, None)
    x = (x*fx)/(z)+x0
    y = (y*fy)/(z)+y0
    xy = np.stack([x, y], axis=-1)
    return xy

def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in list(range(B)):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in list(range(S)):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def apply_4x4_py(RT, xyz):
    # print('RT', RT.shape)
    B, N, _ = list(xyz.shape)
    ones = np.ones_like(xyz[:,:,0:1])
    xyz1 = np.concatenate([xyz, ones], 2)
    # print('xyz1', xyz1.shape)
    xyz1_t = xyz1.transpose(0,2,1)
    # print('xyz1_t', xyz1_t.shape)
    # this is B x 4 x N
    xyz2_t = np.matmul(RT, xyz1_t)
    # print('xyz2_t', xyz2_t.shape)
    xyz2 = xyz2_t.transpose(0,2,1)
    # print('xyz2', xyz2.shape)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def apply_3x3(RT, xy):
    B, N, _ = list(xy.shape)
    ones = torch.ones_like(xy[:,:,0:1])
    xy1 = torch.cat([xy, ones], 2)
    xy1_t = torch.transpose(xy1, 1, 2)
    # this is B x 4 x N
    xy2_t = torch.matmul(RT, xy1_t)
    xy2 = torch.transpose(xy2_t, 1, 2)
    xy2 = xy2[:,:,:2]
    return xy2

def generate_polygon(ctr_x, ctr_y, avg_r, irregularity, spikiness, num_verts):
    '''
    Start with the center of the polygon at ctr_x, ctr_y, 
    Then creates the polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
        ctr_x, ctr_y - coordinates of the "centre" of the polygon
        avg_r - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
        irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
        spikiness - [0,1] indicating how much variance there is in each vertex from the circle of radius avg_r. [0,1] will map to [0, avg_r]
pp        num_verts

    Returns:
        np.array [num_verts, 2] - CCW order.
    '''
    # spikiness
    spikiness = np.clip(spikiness, 0, 1) * avg_r

    # generate n angle steps
    irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / num_verts
    lower = (2*np.pi / num_verts) - irregularity
    upper = (2*np.pi / num_verts) + irregularity

    # angle steps
    angle_steps = np.random.uniform(lower, upper, num_verts)
    sc = (2 * np.pi) / angle_steps.sum()
    angle_steps *= sc

    # get all radii
    angle = np.random.uniform(0, 2*np.pi)
    radii = np.clip(np.random.normal(avg_r, spikiness, num_verts), 0, 2 * avg_r)

    # compute all points
    points = []
    for i in range(num_verts):
        x = ctr_x + radii[i] * np.cos(angle)
        y = ctr_y + radii[i] * np.sin(angle)
        points.append([x, y])
        angle += angle_steps[i]

    return np.array(points).astype(int)


def get_random_affine_2d(B, rot_min=-5.0, rot_max=5.0, tx_min=-0.1, tx_max=0.1, ty_min=-0.1, ty_max=0.1, sx_min=-0.05, sx_max=0.05, sy_min=-0.05, sy_max=0.05, shx_min=-0.05, shx_max=0.05, shy_min=-0.05, shy_max=0.05):
    '''
    Params:
        rot_min: rotation amount min
        rot_max: rotation amount max

        tx_min: translation x min
        tx_max: translation x max

        ty_min: translation y min
        ty_max: translation y max

        sx_min: scaling x min
        sx_max: scaling x max

        sy_min: scaling y min
        sy_max: scaling y max

        shx_min: shear x min
        shx_max: shear x max

        shy_min: shear y min
        shy_max: shear y max

    Returns:
        transformation matrix: (B, 3, 3)
    '''
    # rotation
    if rot_max - rot_min != 0:
        rot_amount = np.random.uniform(low=rot_min, high=rot_max, size=B)
        rot_amount = np.pi/180.0*rot_amount
    else:
        rot_amount = rot_min
    rotation = np.zeros((B, 3, 3)) # B, 3, 3
    rotation[:, 2, 2] = 1
    rotation[:, 0, 0] = np.cos(rot_amount)
    rotation[:, 0, 1] = -np.sin(rot_amount)
    rotation[:, 1, 0] = np.sin(rot_amount)
    rotation[:, 1, 1] = np.cos(rot_amount)

    # translation
    translation = np.zeros((B, 3, 3)) # B, 3, 3
    translation[:, [0,1,2], [0,1,2]] = 1 
    if (tx_max - tx_min) > 0:
        trans_x = np.random.uniform(low=tx_min, high=tx_max, size=B)
        translation[:, 0, 2] = trans_x
    # else:
    #     translation[:, 0, 2] = tx_max
    if ty_max - ty_min != 0:
        trans_y = np.random.uniform(low=ty_min, high=ty_max, size=B)
        translation[:, 1, 2] = trans_y
    # else:
    #     translation[:, 1, 2] = ty_max

    # scaling
    scaling = np.zeros((B, 3, 3)) # B, 3, 3
    scaling[:, [0,1,2], [0,1,2]] = 1 
    if (sx_max - sx_min) > 0:
        scale_x = 1 + np.random.uniform(low=sx_min, high=sx_max, size=B)
        scaling[:, 0, 0] = scale_x
    # else:
    #     scaling[:, 0, 0] = sx_max
    if (sy_max - sy_min) > 0:
        scale_y = 1 + np.random.uniform(low=sy_min, high=sy_max, size=B)
        scaling[:, 1, 1] = scale_y
    # else:
    #     scaling[:, 1, 1] = sy_max

    # shear
    shear = np.zeros((B, 3, 3)) # B, 3, 3
    shear[:, [0,1,2], [0,1,2]] = 1 
    if (shx_max - shx_min) > 0:
        shear_x = np.random.uniform(low=shx_min, high=shx_max, size=B)
        shear[:, 0, 1] = shear_x
    # else:
    #     shear[:, 0, 1] = shx_max
    if (shy_max - shy_min) > 0:
        shear_y = np.random.uniform(low=shy_min, high=shy_max, size=B)
        shear[:, 1, 0] = shear_y
    # else:
    #     shear[:, 1, 0] = shy_max

    # compose all those
    rt = np.einsum("ijk,ikl->ijl", rotation, translation)
    ss = np.einsum("ijk,ikl->ijl", scaling, shear)
    trans = np.einsum("ijk,ikl->ijl", rt, ss)

    return trans

def get_centroid_from_box2d(box2d):
    ymin = box2d[:,0]
    xmin = box2d[:,1]
    ymax = box2d[:,2]
    xmax = box2d[:,3]
    x = (xmin+xmax)/2.0
    y = (ymin+ymax)/2.0
    return y, x

def normalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin = ymin / float(H)
    ymax = ymax / float(H)
    xmin = xmin / float(W)
    xmax = xmax / float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin = ymin * float(H)
    ymax = ymax * float(H)
    xmin = xmin * float(W)
    xmax = xmax * float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_box2d(box2d, H, W):
    return unnormalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)

def normalize_box2d(box2d, H, W):
    return normalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)

def get_size_from_box2d(box2d):
    ymin = box2d[:,0]
    xmin = box2d[:,1]
    ymax = box2d[:,2]
    xmax = box2d[:,3]
    height = ymax-ymin
    width = xmax-xmin
    return height, width

def crop_and_resize(im, boxlist, PH, PW, boxlist_is_normalized=False):
    B, C, H, W = im.shape
    B2, N, D = boxlist.shape
    assert(B==B2)
    assert(D==4)
    # PH, PW is the size to resize to

    # output is B,N,C,PH,PW

    # pt wants xy xy, unnormalized
    if boxlist_is_normalized:
        boxlist_unnorm = unnormalize_boxlist2d(boxlist, H, W)
    else:
        boxlist_unnorm = boxlist
        
    ymin, xmin, ymax, xmax = boxlist_unnorm.unbind(2)
    # boxlist_pt = torch.stack([boxlist_unnorm[:,1], boxlist_unnorm[:,0], boxlist_unnorm[:,3], boxlist_unnorm[:,2]], dim=1)
    boxlist_pt = torch.stack([xmin, ymin, xmax, ymax], dim=2)
    # we want a B-len list of K x 4 arrays

    # print('im', im.shape)
    # print('boxlist', boxlist.shape)
    # print('boxlist_pt', boxlist_pt.shape)

    # boxlist_pt = list(boxlist_pt.unbind(0))

    crops = []
    for b in range(B):
        crops_b = ops.roi_align(im[b:b+1], [boxlist_pt[b]], output_size=(PH, PW))
        crops.append(crops_b)
    # # crops = im

    # print('crops', crops.shape)
    # crops = crops.reshape(B,N,C,PH,PW)

    
    # crops = []
    # for b in range(B):
    #     crop_b = ops.roi_align(im[b:b+1], [boxlist_pt[b]], output_size=(PH, PW))
    #     print('crop_b', crop_b.shape)
    #     crops.append(crop_b)
    crops = torch.stack(crops, dim=0)
        
    # print('crops', crops.shape)
    # boxlist_list = boxlist_pt.unbind(0)
    # print('rgb_crop', rgb_crop.shape)

    return crops


# def get_boxlist_from_centroid_and_size(cy, cx, h, w, clip=True):
#     # cy,cx are both B,N
#     ymin = cy - h/2
#     ymax = cy + h/2
#     xmin = cx - w/2
#     xmax = cx + w/2

#     box = torch.stack([ymin, xmin, ymax, xmax], dim=-1)
#     if clip:
#         box = torch.clamp(box, 0, 1)
#     return box


def get_boxlist_from_centroid_and_size(cy, cx, h, w):#, clip=False):
    # cy,cx are the same shape
    ymin = cy - h/2
    ymax = cy + h/2
    xmin = cx - w/2
    xmax = cx + w/2

    # if clip:
    #     ymin = torch.clamp(ymin, 0, H-1)
    #     ymax = torch.clamp(ymax, 0, H-1)
    #     xmin = torch.clamp(xmin, 0, W-1)
    #     xmax = torch.clamp(xmax, 0, W-1)
    
    box = torch.stack([ymin, xmin, ymax, xmax], dim=-1)
    return box


def get_box2d_from_mask(mask, normalize=False):
    # mask is B, 1, H, W

    B, C, H, W = mask.shape
    assert(C==1)
    xy = utils.basic.gridcloud2d(B, H, W, norm=False, device=mask.device) # B, H*W, 2

    box = torch.zeros((B, 4), dtype=torch.float32, device=mask.device)
    for b in range(B):
        xy_b = xy[b] # H*W, 2
        mask_b = mask[b].reshape(H*W)
        xy_ = xy_b[mask_b > 0]
        x_ = xy_[:,0]
        y_ = xy_[:,1]
        ymin = torch.min(y_)
        ymax = torch.max(y_)
        xmin = torch.min(x_)
        xmax = torch.max(x_)
        box[b] = torch.stack([ymin, xmin, ymax, xmax], dim=0)
    if normalize:
        box = normalize_boxlist2d(box.unsqueeze(1), H, W).squeeze(1)
    return box

def convert_box2d_to_intrinsics(box2d, pix_T_cam, H, W, use_image_aspect_ratio=True, mult_padding=1.0):
    # box2d is B x 4, with ymin, xmin, ymax, xmax in normalized coords
    # ymin, xmin, ymax, xmax = torch.unbind(box2d, dim=1)
    # H, W is the original size of the image
    # mult_padding is relative to object size in pixels

    # i assume we're rendering an image the same size as the original (H, W)

    if not mult_padding==1.0:
        y, x = get_centroid_from_box2d(box2d)
        h, w = get_size_from_box2d(box2d)
        box2d = get_box2d_from_centroid_and_size(
            y, x, h*mult_padding, w*mult_padding, clip=False)
        
    if use_image_aspect_ratio:
        h, w = get_size_from_box2d(box2d)
        y, x = get_centroid_from_box2d(box2d)

        # note h,w are relative right now
        # we need to undo this, to see the real ratio

        h = h*float(H)
        w = w*float(W)
        box_ratio = h/w
        im_ratio = H/float(W)

        # print('box_ratio:', box_ratio)
        # print('im_ratio:', im_ratio)

        if box_ratio >= im_ratio:
            w = h/im_ratio
            # print('setting w:', h/im_ratio)
        else:
            h = w*im_ratio
            # print('setting h:', w*im_ratio)
            
        box2d = get_box2d_from_centroid_and_size(
            y, x, h/float(H), w/float(W), clip=False)

    assert(h > 1e-4)
    assert(w > 1e-4)
        
    ymin, xmin, ymax, xmax = torch.unbind(box2d, dim=1)

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)

    # the topleft of the new image will now have a different offset from the center of projection
    
    new_x0 = x0 - xmin*W
    new_y0 = y0 - ymin*H

    pix_T_cam = pack_intrinsics(fx, fy, new_x0, new_y0)
    # this alone will give me an image in original resolution,
    # with its topleft at the box corner

    box_h, box_w = get_size_from_box2d(box2d)
    # these are normalized, and shaped B. (e.g., [0.4], [0.3])

    # we are going to scale the image by the inverse of this,
    # since we are zooming into this area

    sy = 1./box_h
    sx = 1./box_w

    pix_T_cam = scale_intrinsics(pix_T_cam, sx, sy)
    return pix_T_cam, box2d

def pixels2camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth in meters
    # they can be images or pointclouds
    # fx, fy, x0, y0 are camera intrinsics
    # returns xyz, sized B x N x 3

    B = x.shape[0]
    
    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])

    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    
    # unproject
    x = (z/fx)*(x-x0)
    y = (z/fy)*(y-y0)
    
    xyz = torch.stack([x,y,z], dim=2)
    # B x N x 3
    return xyz

def camera2pixels(xyz, pix_T_cam):
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/z + x0
    y = (y*fy)/z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy

def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    device = z.device
    y, x = utils.basic.meshgrid2d(B, H, W, device=device)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(x, y, z, fx, fy, x0, y0)
    return xyz
