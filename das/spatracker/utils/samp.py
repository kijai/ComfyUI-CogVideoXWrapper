import torch
import utils.basic
import torch.nn.functional as F

def bilinear_sample2d(im, x, y, return_inbounds=False):
    # x and y are each B, N
    # output is B, C, N
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)
    
    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    
    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device)*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    im_flat = (im.permute(0, 2, 3, 1)).reshape(B*H*W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + \
             w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(B, N) # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output # B, C, N

def paste_crop_on_canvas(crop, box2d_unnorm, H, W, fast=True, mask=None, canvas=None):
    # this is the inverse of crop_and_resize_box2d
    B, C, Y, X = list(crop.shape)
    B2, D = list(box2d_unnorm.shape)
    assert(B == B2)
    assert(D == 4)

    # here, we want to place the crop into a bigger image,
    # at the location specified by the box2d.

    if canvas is None:
        canvas = torch.zeros((B, C, H, W), device=crop.device)
    else:
        B2, C2, H2, W2 = canvas.shape
        assert(B==B2)
        assert(C==C2)
        assert(H==H2)
        assert(W==W2)

    # box2d_unnorm = utils.geom.unnormalize_box2d(box2d, H, W)

    if fast:
        ymin = box2d_unnorm[:, 0].long()
        xmin = box2d_unnorm[:, 1].long()
        ymax = box2d_unnorm[:, 2].long()
        xmax = box2d_unnorm[:, 3].long()
        w = (xmax - xmin).float()
        h = (ymax - ymin).float()

        grids = utils.basic.gridcloud2d(B, H, W)
        grids_flat = grids.reshape(B, -1, 2)
        # grids_flat[:, :, 0] = (grids_flat[:, :, 0] - xmin.float().unsqueeze(1)) / w.unsqueeze(1) * X
        # grids_flat[:, :, 1] = (grids_flat[:, :, 1] - ymin.float().unsqueeze(1)) / h.unsqueeze(1) * Y

        # for each pixel in the main image,
        # grids_flat tells us where to sample in the crop image

        # print('grids_flat', grids_flat.shape)
        # print('crop', crop.shape)

        grids_flat[:, :, 0] = (grids_flat[:, :, 0] - xmin.float().unsqueeze(1)) / w.unsqueeze(1) * 2.0 - 1.0
        grids_flat[:, :, 1] = (grids_flat[:, :, 1] - ymin.float().unsqueeze(1)) / h.unsqueeze(1) * 2.0 - 1.0
        
        grid = grids_flat.reshape(B,H,W,2)

        canvas = F.grid_sample(crop, grid, align_corners=False)
        # print('canvas', canvas.shape)
        
        # if mask is None:
        #     crop_resamp, inb = bilinear_sample2d(crop, grids_flat[:, :, 0], grids_flat[:, :, 1], return_inbounds=True)
        #     crop_resamp = crop_resamp.reshape(B, C, H, W)
        #     inb = inb.reshape(B, 1, H, W)
        #     canvas = canvas * (1 - inb) + crop_resamp * inb
        # else:
        #     full_resamp = bilinear_sample2d(torch.cat([crop, mask], dim=1), grids_flat[:, :, 0], grids_flat[:, :, 1])
        #     full_resamp = full_resamp.reshape(B, C+1, H, W)
        #     crop_resamp = full_resamp[:,:3]
        #     mask_resamp = full_resamp[:,3:4]
        #     canvas = canvas * (1 - mask_resamp) + crop_resamp * mask_resamp
    else:
        for b in range(B):
            ymin = box2d_unnorm[b, 0].long()
            xmin = box2d_unnorm[b, 1].long()
            ymax = box2d_unnorm[b, 2].long()
            xmax = box2d_unnorm[b, 3].long()

            crop_b = F.interpolate(crop[b:b + 1], (ymax - ymin, xmax - xmin)).squeeze(0)

            # print('canvas[b,:,...', canvas[b,:,ymin:ymax,xmin:xmax].shape)
            # print('crop_b', crop_b.shape)

            canvas[b, :, ymin:ymax, xmin:xmax] = crop_b
    return canvas
