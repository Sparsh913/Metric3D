import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os, cv2
import numpy as np
from mono.utils.transform import gray_to_colormap
import shutil
import glob
from mono.utils.running import main_process
import torch
from html4vision import Col, imagetable

def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    if value.ndim > 2:
        return value
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[..., :3]
    return img

def save_raw_imgs( 
    pred: torch.tensor,  
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str,
    scale: float=200.0, 
    target: torch.tensor=None,
    ):
    """
    Save raw GT, predictions, RGB in the same file.
    """
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), rgb)
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_d.png'), (pred*scale).astype(np.uint16))
    if target is not None:
        cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_gt.png'), (target*scale).astype(np.uint16))
    

def save_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    target: torch.tensor,
    rgb: torch.tensor, 
    rmse: torch.tensor,
    filename: str, 
    save_dir: str, 
    tb_logger=None
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    rgb, pred_scale, target_scale, pred_color, target_color, rmse_color = get_data_for_log(pred, target, rgb, rmse)
    rgb = rgb.transpose((1, 2, 0))
    top_row = np.concatenate([rgb, pred_color], axis=1)
    bottom_row = np.concatenate([target_color, rmse_color], axis=1)
    # cat_img = np.concatenate([rgb, pred_color, target_color], axis=0)
    cat_img = np.concatenate([top_row, bottom_row], axis=0)
    plt.imsave(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), cat_img)

    # save to tensorboard
    if tb_logger is not None:
        tb_logger.add_image(f'{filename[:-4]}_merge.jpg', cat_img.transpose((2, 0, 1)), iter)

def save_normal_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    targ: torch.tensor, 
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    tb_logger=None, 
    mask=None,
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    mean = np.array([123.675, 116.28, 103.53])[np.newaxis, np.newaxis, :]
    std= np.array([58.395, 57.12, 57.375])[np.newaxis, np.newaxis, :]
    pred = pred.squeeze()
    targ = targ.squeeze()
    rgb = rgb.squeeze()

    if pred.size(0) == 3:
        pred = pred.permute(1,2,0)
    if targ.size(0) == 3:
        targ = targ.permute(1,2,0)
    if rgb.size(0) == 3:
        rgb = rgb.permute(1,2,0)

    pred_color = vis_surface_normal(pred, mask)
    targ_color = vis_surface_normal(targ, mask)
    rgb_color = ((rgb.cpu().numpy() * std) + mean).astype(np.uint8)

    try:
        cat_img = np.concatenate([rgb_color, pred_color, targ_color], axis=0)
    except:
        pred_color = cv2.resize(pred_color, (rgb.shape[1], rgb.shape[0]))
        targ_color = cv2.resize(targ_color, (rgb.shape[1], rgb.shape[0]))
        cat_img = np.concatenate([rgb_color, pred_color, targ_color], axis=0)

    plt.imsave(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), cat_img)
    # cv2.imwrite(os.path.join(save_dir, filename[:-4]+'.jpg'), pred_color)
    # save to tensorboard
    if tb_logger is not None:
        tb_logger.add_image(f'{filename[:-4]}_merge.jpg', cat_img.transpose((2, 0, 1)), iter)

def get_data_for_log(pred: torch.tensor, target: torch.tensor, rgb: torch.tensor, rmse: torch.tensor = None):
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()
    rmse = rmse.squeeze().cpu().numpy()

    pred[pred<0] = 0
    target[target<0] = 0
    max_scale = max(pred.max(), target.max())
    rmse_max = rmse.max()
    pred_scale = (pred/max_scale * 10000).astype(np.uint16)
    target_scale = (target/max_scale * 10000).astype(np.uint16)
    rmse_scale = (rmse/rmse_max * 10000).astype(np.uint16)
    pred_color = gray_to_colormap(pred)
    target_color = gray_to_colormap(target)
    rmse_color = rmse_colormap(rmse)
    pred_color = cv2.resize(pred_color, (rgb.shape[2], rgb.shape[1]))
    target_color = cv2.resize(target_color, (rgb.shape[2], rgb.shape[1]))
    rmse_color = cv2.resize(rmse_color, (rgb.shape[2], rgb.shape[1]))

    rgb = ((rgb * std) + mean).astype(np.uint8)
    return rgb, pred_scale, target_scale, pred_color, target_color, rmse_color


def create_html(name2path, save_path='index.html', size=(256, 384)):
    # table description
    cols = []
    for k, v in name2path.items():
        col_i =  Col('img', k, v) # specify image content for column
        cols.append(col_i)
    # html table generation
    imagetable(cols, out_file=save_path, imsize=size)

def vis_surface_normal(normal: torch.tensor, mask: torch.tensor=None) -> np.array:
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    Aargs:
        normal (torch.tensor, [h, w, 3]): surface normal
        mask (torch.tensor, [h, w]): valid masks
    """
    normal = normal.cpu().numpy().squeeze()
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    if mask is not None:
        mask = mask.cpu().numpy().squeeze()
        normal_vis[~mask] = 0
    return normal_vis

def rmse_colormap(img, cmap='rainbow', min_val=0, max_val=80.0):
    assert img.ndim == 2

    # Ensure values are within the expected range
    img[img < 0] = 0
    img[img > max_val] = max_val

    # Normalize based on absolute min and max values
    img_normalized = (img - min_val) / (max_val - min_val + 1e-8)
    img_normalized = np.clip(img_normalized, 0, 1)  # Ensure values are in [0, 1]

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    
    # Apply colormap
    colormap = (map.to_rgba(img_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Set invalid regions to black (or any color you want)
    mask_invalid = img < 1e-10
    colormap[mask_invalid] = 0  # Set invalid regions to black

    return colormap

def save_val_imgs_v2(
    iter: int, 
    depth_pred: torch.tensor, 
    depth_gt: torch.tensor,
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    tb_logger=None,
    active_mask: torch.tensor=None,
    valid_depth_mask: torch.tensor=None,
    depth_max=20,
    arel_max=0.3
    ):
    
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    depth_pred = depth_pred.squeeze().cpu().numpy()
    depth_gt = depth_gt.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()
    rgb = ((rgb * std) + mean).astype(np.uint8)
    if active_mask is not None:
        active_mask = active_mask.squeeze().cpu().numpy()
        rgb = rgb * (active_mask>0).astype(np.uint8)
        depth_pred = depth_pred * (active_mask>0).astype(np.float32)
    if valid_depth_mask is not None:
        valid_depth_mask = valid_depth_mask.squeeze().cpu().bool().numpy()
        depth_gt = depth_gt * valid_depth_mask
    rgb = rgb.transpose((1, 2, 0))
    
    # compute error, you have zero divison where depth_gt == 0.0
    depth_arel = np.abs(depth_gt - depth_pred) / (depth_gt + 1e-9)
    depth_arel[depth_gt == 0.0] = 0.0
    
    cmap_depth = cm.magma_r  # Use any colormap you like (e.g., 'viridis', 'plasma', 'inferno', etc.)
    norm_depth = mcolors.Normalize(vmin=0, vmax=depth_max)  # Set the data range for the color bar
    cmap_arel = cm.coolwarm  # Use any colormap you like (e.g., 'viridis', 'plasma', 'inferno', etc.)
    norm_arel = mcolors.Normalize(vmin=0, vmax=arel_max)  # Set the data range for the color bar

    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(rgb)
    # plt.title("Image")
    
    # plt.subplot(2, 2, 2)
    # ax=plt.imshow(depth_arel, cmap=cmap_arel, norm=norm_arel)
    # plt.colorbar(ax, label="A.Rel")
    # plt.title("A. Rel")
    
    # plt.subplot(2, 2, 3)
    # ax = plt.imshow(depth_gt, cmap=cmap_depth, norm=norm_depth)
    # plt.colorbar(ax, label="Meter")
    # plt.title("Depth GT")
    
    # plt.subplot(2, 2, 4)
    # ax = plt.imshow(depth_pred, cmap=cmap_depth, norm=norm_depth)
    # plt.colorbar(ax, label="Meter")
    # plt.title("Depth Pred")
    
    # plt.savefig(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), dpi=200)
    
    # save all the subplots as individual images
    plt.figure()
    plt.imshow(rgb)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    # plt.title("Image")
    plt.savefig(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), dpi=200, bbox_inches='tight', pad_inches=0)
    
    plt.figure()
    plt.imshow(depth_arel, cmap=cmap_arel, norm=norm_arel)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    # No need to save colorbar as it is saved in the main image
    # plt.colorbar(ax, label="A.Rel")
    # plt.title("A. Rel")
    plt.savefig(os.path.join(save_dir, filename[:-4]+'_arel.jpg'), dpi=200, bbox_inches='tight', pad_inches=0)
    
    plt.figure()
    ax = plt.imshow(depth_gt, cmap=cmap_depth, norm=norm_depth)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(os.path.join(save_dir, filename[:-4]+'_gt.jpg'), dpi=200, bbox_inches='tight', pad_inches=0)
    
    plt.figure()
    ax = plt.imshow(depth_pred, cmap=cmap_depth, norm=norm_depth)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(os.path.join(save_dir, filename[:-4]+'_pred.jpg'), dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close('all')