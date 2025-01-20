import os
import numpy as np
import re
import yaml
import sys

import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

"""

    Grid search method to prepare fisheye grid for NYU dataset.
    Ignoring the tangential distortion terms which causes little impact.
    Simply treating the data as perspective images are also OK, the overall distortion is not that severe.
    
"""

cam_params = {
    'dataset': 'nyu',
    'fx': 5.1885790117450188e+02,
    'fy': 5.1946961112127485e+02,
    'cx': 3.2558244941119034e+02,
    'cy': 2.5373616633400465e+02,
    'k1': 2.0796615318809061e-01,
    'k2': -5.8613825163911781e-01,
    'p1': 7.2231363135888329e-04,
    'p2': 1.0479627195765181e-03,
    'k3': 4.9856986684705107e-01,
    'wFOV': 1.105,
    'hFOV': 0.8663
    }

if __name__=="__main__":
    import cv2
    import matplotlib.pyplot as plt

    NYUPath = 'datasets/nyu/'

    H = 480
    W = 640
    # [H, W]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # [H*W]
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5

    # [3, H*W]
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)

    grid = torch.from_numpy(pixels)

    map_dist = []
    z_dist = []
    
    k1 = cam_params['k1']
    k2 = cam_params['k2']
    k3 = cam_params['k3']
    p1 = cam_params['p1']
    p2 = cam_params['p2']
    fx = cam_params['fx']
    fy = cam_params['fy']
    cx = cam_params['cx']
    cy = cam_params['cy']

    # ATTENTION, the range of ro2 can go beyond 1.0 for fisheye cameras large FoV, set it properly
    for ro2 in np.linspace(0.0, 1.0, 200000):
        ro2_after = np.sqrt(ro2) * (1 + k1*ro2 + k2*ro2*ro2 + k3*ro2*ro2*ro2)
        map_dist.append([(1 + k1*ro2 + k2*ro2*ro2 + k3*ro2*ro2*ro2), ro2_after])
    map_dist = np.array(map_dist).astype(np.float32)
    # print(map_dist)
    map_dist = torch.from_numpy(map_dist).cuda()
    
    def chunk(grid):
        x = grid[0, :]
        y = grid[1, :]

        x = (x - cx) / fx
        y = (y - cy) / fy
        dist = torch.sqrt(x*x + y*y)
        indx = torch.abs(map_dist[:, 1:] - dist[None, :]).argmin(dim=0)
        x /= map_dist[indx, 0]
        y /= map_dist[indx, 0]
        
        # z has closed form solution sqrt(1 - z^2) / z = sqrt(x^2 + y^2)
        z = 1 / torch.sqrt(1 + x**2 + y**2)
    
        xy = torch.stack((x, y))
        xy *= z
        return xy

    xys = []
    for i in range(H):
        xy = chunk(grid[:, i*W:(i+1)*W].cuda())
        xys.append(xy.permute(1, 0))
        if i % 10 == 0:
            print(i)
    xys = torch.cat(xys, dim=0)

    z = torch.sqrt(1. - torch.norm(xys, dim=1, p=2) ** 2)
    isnan = z.isnan()
    z[isnan] = 1.
    pcd = torch.cat((xys, z[:, None], isnan[:, None]), dim=1)
    print("saving grid")
    np.save(os.path.join('splits/nyu', f'grid_fisheye.npy'), pcd.detach().cpu().numpy().reshape(H, W, 4))

    """
        Treating each ray as a point on an unit sphere, apply forward distortion and project to compute the approximation error using the lookup table
    """
    def cam2image(pcd):
        x = pcd[:, 0] / pcd[:, 2]
        y = pcd[:, 1] / pcd[:, 2]
        z = pcd[:, 2]
        
        r_sq = x**2 + y**2
        dx = 2*p1*x*y + p2*(r_sq + 2*x**2)
        dy = p1*(r_sq + 2*y**2) + 2*p2*x*y
        x = (1 + k1*r_sq + k2*r_sq**2 + k3*r_sq**3)*x + dx
        y = (1 + k1*r_sq + k2*r_sq**2 + k3*r_sq**3)*y + dy
        # x = (1 + k1*r_sq + k2*r_sq**2 + k3*r_sq**3)*x
        # y = (1 + k1*r_sq + k2*r_sq**2 + k3*r_sq**3)*y     

        """
            Projection to image coordinates using intrinsic parameters
        """
        x = fx * x + cx
        y = fy * y + cy

        return x, y, z
    
    # import ipdb; ipdb.set_trace()
    # show error map
    x, y, d = cam2image(pcd[:, :3])

    error = (x - grid[0].cuda()) ** 2 + (y - grid[1].cuda()) ** 2

    error_map = error.reshape(H, W).detach().cpu().numpy()
    error_map = np.clip(error_map, 0, 30)
    plt.imshow(error_map)
    plt.show()

