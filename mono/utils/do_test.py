import torch
import torch.nn.functional as F
import logging
import os
import os.path as osp
from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, create_html, save_raw_imgs, save_normal_val_imgs, save_val_imgs_v2
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import OpenEXR
# import Imath

from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud, reconstruct_pcd_erp
from mono.utils.comm import main_process
from mono.utils.erp_geometry import erp_patch_to_cam_fast
from torch.utils.data import DataLoader


def to_cuda(data: dict):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v)>=1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data

def align_scale(pred: torch.tensor, target: torch.tensor):
    mask = (target > 0) * (target < 40)
    if torch.sum(mask) > 10:
        scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
    else:
        scale = 1
    pred_scaled = pred * scale
    return pred_scaled, scale

def align_scale_shift(pred: torch.tensor, target: torch.tensor):
    mask = (target > 0) * (target < 40)
    target_mask = target[mask].cpu().numpy()
    pred_mask = pred[mask].cpu().numpy()
    if torch.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale

def align_scale_shift_numpy(pred: np.array, target: np.array):
    mask = (target > 0) * (target < 40)
    target_mask = target[mask]
    pred_mask = pred[mask]
    if np.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale


def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map. 
    """
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - np.array(u0)) / W
    x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - np.array(v0)) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T # [H, W]

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y = np.arctan(y_center / (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model

def resize_for_input(image, output_shape, intrinsic, canonical_shape, to_canonical_ratio, mask=None):
    """
    Resize the input.
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    """
    padding = [123.675, 116.28, 103.53]
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    # padding
    image = cv2.copyMakeBorder(
        image, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=padding)
    
    # Resize, adjust principle point
    intrinsic[2] = intrinsic[2] * to_scale_ratio
    intrinsic[3] = intrinsic[3] * to_scale_ratio

    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = cv2.copyMakeBorder(
        cam_model, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=-1)

    pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor=1/to_scale_ratio
    if mask is not None:
        mask = cv2.resize(mask, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.copyMakeBorder(
            mask, 
            pad_h_half, 
            pad_h - pad_h_half, 
            pad_w_half, 
            pad_w - pad_w_half, 
            cv2.BORDER_CONSTANT, 
            value=0)
        return image, cam_model, pad, label_scale_factor, mask
    return image, cam_model, pad, label_scale_factor


def get_prediction(
    model: torch.nn.Module,
    input: torch.tensor,
    cam_model: torch.tensor,
    pad_info: torch.tensor,
    scale_info: torch.tensor,
    gt_depth: torch.tensor,
    normalize_scale: float,
    ori_shape: list=[],
):

    data = dict(
        input=input,
        cam_model=cam_model,
    )
    pred_depth, confidence, output_dict = model.module.inference(data)
    # pred_depth_scale, scale = align_scale(pred_depth, gt_depth)

    return pred_depth, output_dict

def transform_test_data_scalecano(rgb, intrinsic, data_basic):
    """
    Pre-process the input for forwarding. Employ `label scale canonical transformation.'
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    canonical_space = data_basic['canonical_space']
    forward_size = data_basic.crop_size
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    # BGR to RGB
    #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space['focal_length']

    cano_label_scale_ratio = canonical_focal / ori_focal

    canonical_intrinsic = [
        intrinsic[0] * cano_label_scale_ratio,
        intrinsic[1] * cano_label_scale_ratio,
        intrinsic[2],
        intrinsic[3],
    ]

    # resize
    # if active_mask is not None:
    #     rgb, cam_model, pad, resize_label_scale_ratio, attn_mask = resize_for_input(rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0, active_mask)
    # else:
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0)

    # label scale factor
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb.cuda()
    
    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].cuda()
    cam_model_stacks = [
        torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
        for i in [2, 4, 8, 16, 32]
    ]
    # if active_mask is not None:  
    #     return rgb, cam_model_stacks, pad, label_scale_factor, attn_mask
    return rgb, cam_model_stacks, pad, label_scale_factor

# def read_exr(filename):
#     exr_file = OpenEXR.InputFile(filename)
#     header = exr_file.header()
#     dw = header['dataWindow']
#     size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
#     # Assumes depth channel is named "Z"
#     depth_str = exr_file.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
#     depth = np.frombuffer(depth_str, dtype=np.float32)
#     depth.shape = (size[1], size[0])  # Reshape to 2D array

#     return depth

def do_scalecano_test_with_custom_data(
    model: torch.nn.Module,
    cfg: dict,
    test_data: list,
    # valid_loader: DataLoader,
    logger: logging.RootLogger,
    is_distributed: bool = True,
    local_rank: int = 0,
    bs: int = 2,  # Batch size parameter
):

    show_dir = cfg.show_dir
    save_interval = 1
    save_imgs_dir = show_dir + '/vis'
    os.makedirs(save_imgs_dir, exist_ok=True)
    save_pcd_dir = show_dir + '/pcd'
    os.makedirs(save_pcd_dir, exist_ok=True)

    normalize_scale = cfg.data_basic.depth_range[1]
    dam = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3', 'log10', 'rmse_log'])
    dam_median = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3', 'log10', 'rmse_log'])
    dam_global = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3', 'log10', 'rmse_log'])
    
    # Process data in batches
    # for i, batch in enumerate(tqdm(valid_loader)):
    for i in range(0, len(test_data), bs):
        batch_data = test_data[i:i + bs]  # Extract batch
        rgb_inputs, pads, label_scale_factors, gt_depths, rgb_origins, attn_masks = [], [], [], [], [], []
        for k, an in enumerate(batch_data):
            rgb_origin = an['rgb'] # tensor
            # print("rgb_origin shape", rgb_origin.shape)
            # convert tensor to numpy
            rgb_origin = rgb_origin.permute(1, 2, 0).cpu().numpy()
            rgb_origins.append(rgb_origin)
            gt_depth = an['depth'] # tensor
            # print("gt_depth shape", gt_depth.shape)
            gt_depth = gt_depth.permute(1, 2, 0).cpu().numpy()
            gt_depths.append(gt_depth)
            intrinsic = an['intrinsic']
            print("intrinsic", intrinsic)
            if intrinsic is None:
                intrinsic = [1000.0, 1000.0, rgb_origin.shape[1]/2, rgb_origin.shape[0]/2]
            # active_mask = an["active_mask"]
            
            # if active_mask is not None:
            #     rgb_input, _, pad, label_scale_factor, attn_mask = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic, active_mask)
            #     attn_masks.append(attn_mask)
            # else:
            rgb_input, _, pad, label_scale_factor= transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic)
            rgb_inputs.append(rgb_input)
            pads.append(pad)
            label_scale_factors.append(label_scale_factor)

        # Process the batch
        pred_depths, outputs = get_prediction(
            model=model,
            input=torch.stack(rgb_inputs),  # Stack inputs for batch processing
            cam_model=None,
            pad_info=pads,
            scale_info=None,
            gt_depth= None,
            normalize_scale=None,
        )
        
        for j, gt_depth in enumerate(gt_depths):
            normal_out = None
            if 'normal_out_list'  in outputs.keys():
                normal_out = outputs['normal_out_list'][0][j, :]
            
            postprocess_per_image(
                i*bs+j,
                pred_depths[j, :],
                gt_depth,
                intrinsic,
                rgb_origins[j],
                normal_out,
                pads[j],
                batch_data[j],
                dam,
                dam_median,
                dam_global,
                is_distributed,
                save_imgs_dir,
                save_pcd_dir,
                normalize_scale,
                label_scale_factors[j],
                cfg,
                # attn_masks[j]
            )

    if gt_depth is not None:
    # if False:
        eval_error = dam.get_metrics()
        print('w/o match :', eval_error)

        eval_error_median = dam_median.get_metrics()
        print('median match :', eval_error_median)

        eval_error_global = dam_global.get_metrics()
        print('global match :', eval_error_global)
    else:
        print('missing gt_depth, only save visualizations...')


def postprocess_per_image(i, pred_depth, gt_depth, intrinsic, rgb_origin, normal_out, pad, an, dam, dam_median, dam_global, is_distributed, save_imgs_dir, save_pcd_dir, normalize_scale, scale_info, cfg, attn_mask=None):
    
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad[0] : pred_depth.shape[0] - pad[1], pad[2] : pred_depth.shape[1] - pad[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], [rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear').squeeze().to("cuda") # to original size
    if isinstance(scale_info, torch.Tensor):
        print("scale_info type", type(scale_info))
        scale_info = scale_info.to("cuda")
    if isinstance(normalize_scale, torch.Tensor):
        print("normalize_scale type", type(normalize_scale))
        normalize_scale = normalize_scale.to("cuda")
    pred_depth = pred_depth * normalize_scale / scale_info

    pred_depth = (pred_depth > 0) * (pred_depth < 300) * pred_depth
    if gt_depth is not None:

        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], (gt_depth.shape[0], gt_depth.shape[1]), mode='bilinear').squeeze() # to original size

        gt_depth = torch.from_numpy(gt_depth).cuda()
        gt_depth = gt_depth.squeeze()
        # print("gt_depth shape", gt_depth.shape)
        # print("pred_depth shape", pred_depth.shape)

        pred_depth_median = pred_depth * gt_depth[gt_depth != 0].median() / pred_depth[gt_depth != 0].median()
        pred_global, _ = align_scale_shift(pred_depth, gt_depth)
        
        mask = (gt_depth > 1e-8) * (gt_depth < 40)
        dam.update_metrics_gpu(pred_depth, gt_depth, mask, is_distributed)
        dam_median.update_metrics_gpu(pred_depth_median, gt_depth, mask, is_distributed)
        dam_global.update_metrics_gpu(pred_global, gt_depth, mask, is_distributed)
        # print(gt_depth[gt_depth != 0].median() / pred_depth[gt_depth != 0].median(), )
    
    # print min, max, mean, median of pred_depth, gt_depth
    # print("pred_depth: min, max, mean, median: ", torch.min(pred_depth), torch.max(pred_depth), torch.mean(pred_depth), torch.median(pred_depth))
    # if gt_depth is not None:
    #     print("gt_depth: min, max, mean, median: ", torch.min(gt_depth), torch.max(gt_depth), torch.mean(gt_depth), torch.median(gt_depth))
    # os.makedirs(osp.join(save_imgs_dir, an['folder']), exist_ok=True)
    # rgb_torch = torch.from_numpy(rgb_origin).to(pred_depth.device).permute(2, 0, 1)
    # mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None].to(rgb_torch.device)
    # std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None].to(rgb_torch.device)
    # rgb_torch = torch.div((rgb_torch - mean), std)
    # rmse = torch.sqrt((pred_depth - gt_depth) ** 2)
    # valid_mask = (gt_depth > 1e-8) * (gt_depth < 80)
    # rmse = torch.where(valid_mask, rmse, torch.tensor(0.0).to(rmse.device))
    # # print("gt_depth shape: ", gt_depth.shape)
    # # print("percent nonzero pixels", torch.count_nonzero(gt_depth) / gt_depth.numel())

    # # save_val_imgs(
    # #     i,
    # #     pred_depth,
    # #     gt_depth if gt_depth is not None else torch.ones_like(pred_depth, device=pred_depth.device),
    # #     rgb_torch,
    # #     rmse,
    # #     osp.join(an['folder'], an['filename']),
    # #     save_imgs_dir,
    # # )
    # # active_mask = np.logical_and(np.logical_and(u_d > 0, u_d < input_w), np.logical_and(v_d > 0, v_d < input_h)) * 1.0
    # active_mask = np.logical_and(np.logical_and(rgb_origin[:, :, 0] > 0, rgb_origin[:, :, 0] < rgb_origin.shape[1]), np.logical_and(rgb_origin[:, :, 1] > 0, rgb_origin[:, :, 1] < rgb_origin.shape[0])) * 1.0
    
    # vis_depth_max = 10.0
    # vis_arel_max = 0.50
    # # rgb = save_val_imgs_v2(
    # #     i,
    # #     pred_depth,
    # #     gt_depth if gt_depth is not None else torch.ones_like(pred_depth, device=pred_depth.device),
    # #     rgb_torch,
    # #     f'rgb_{i:06d}.jpg',
    # #     save_imgs_dir,
    # #     active_mask=torch.from_numpy(active_mask) if active_mask is not None else None,
    # #     valid_depth_mask=valid_mask,
    # #     depth_max=vis_depth_max,
    # #     arel_max=vis_arel_max
    # # )
    # p_depth = pred_depth.detach().cpu().numpy()
    # data_dir = '/home/azureuser/cloudfiles/code/Users/gas1syv/metric3d_data/kitti'
    # if "KITTI" in cfg.data_array[0][0]:
    #     fisheye_file = an['rgb']
    #     if "image_02" in fisheye_file:
    #         grid_fisheye = np.load(os.path.join(data_dir, 'fisheye', 'grid_fisheye_02.npy'))
    #         mask_fisheye = np.load(os.path.join(data_dir, 'fisheye', 'mask_left_fisheye.npy'))
    #     elif "image_03" in fisheye_file:
    #         grid_fisheye = np.load(os.path.join(data_dir, 'fisheye', 'grid_fisheye_03.npy'))
    #         mask_fisheye = np.load(os.path.join(data_dir, 'fisheye', 'mask_right_fisheye.npy'))
    #     gits_isnan = cv2.resize(grid_fisheye[:, :, 3], (p_depth.shape[1], p_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    #     grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (p_depth.shape[1], p_depth.shape[0]))
    #     grid_fisheye = np.concatenate([grid_fisheye, gits_isnan[:, :, None]], axis=2)
    #     mask_fisheye = cv2.resize(mask_fisheye.astype(np.uint8), (p_depth.shape[1], p_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    #     img_h, img_w = rgb_torch.shape[1], rgb_torch.shape[2]
    #     img_out, depth_out, valid_mask, active_mask, depth_out_gt = erp_patch_to_cam_fast(
    #         rgb_torch.to("cpu"), pred_depth.unsqueeze(0).detach().cpu(), torch.ones_like(pred_depth.unsqueeze(0)).to("cpu"), 0, 0, out_h=img_h, out_w=img_w, erp_h=img_h, erp_w=img_h*2, cam_params={'dataset':'kitti360'},
    #         fisheye_grid2ray=grid_fisheye, depth_erp_gt=gt_depth.unsqueeze(0).detach().cpu())
    #     rgb = save_val_imgs_v2(
    #         i,
    #         depth_out,
    #         depth_out_gt,
    #         img_out,
    #         f'rgb_{i:06d}_remap_org_cam.jpg',
    #         save_dir=save_imgs_dir,
    #         active_mask=active_mask,
    #         valid_depth_mask=valid_mask,
    #         depth_max=vis_depth_max,
    #         arel_max=vis_arel_max
    #     )
    # else:
    #     """
    #         Currently work perfect with phi = 0. For larger phi, corners may have artifacts.
    #     """
    #     data_dir = '/home/azureuser/cloudfiles/code/Users/gas1syv/metric3d_data/scannet/scannetpp_tiny'
    #     rgb_file = an['rgb']
    #     grid_fisheye = np.load(os.path.join(data_dir, 'data', rgb_file.split('/')[-4], 'dslr', 'grid_fisheye.npy'))
    #     out_h = 500
    #     out_w = 750
    #     gits_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    #     grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
    #     grid_fisheye = np.concatenate([grid_fisheye, gits_isnan[:, :, None]], axis=2)
    #     erp_h = 1400
    #     phi = np.array([0])
    #     # erp_h *= 1/scale_info  #1/np.tan(np.pi/1400)/519.0 #0.4599
    #     # erp_h = erp_h*0.92
    #     cam_params={'dataset':'scannetpp'}
    #     active_mask = np.logical_and(np.logical_and(rgb_origin[:, :, 0] > 0, rgb_origin[:, :, 0] < rgb_origin.shape[1]), np.logical_and(rgb_origin[:, :, 1] > 0, rgb_origin[:, :, 1] < rgb_origin.shape[0])) * 1.0
    #     img_out, depth_out, valid_mask, active_mask, depth_out_gt = erp_patch_to_cam_fast(
    #         rgb_torch.to("cpu"), pred_depth.unsqueeze(0).detach().cpu(), torch.from_numpy(active_mask).unsqueeze(0).to("cpu"), 0., phi, out_h=out_h, out_w=out_w, erp_h=erp_h, erp_w=erp_h*2,
    #         cam_params=cam_params, fisheye_grid2ray=grid_fisheye.astype(np.float32), depth_erp_gt=gt_depth.unsqueeze(0).detach().cpu()
    #     )
    #     rgb = save_val_imgs_v2(
    #         i,
    #         depth_out,
    #         depth_out_gt,
    #         img_out,
    #         f'rgb_{i:06d}_remap_org_cam.jpg',
    #         save_dir=save_imgs_dir,
    #         active_mask=active_mask,
    #         valid_depth_mask=valid_mask,
    #         depth_max=vis_depth_max,
    #         arel_max=vis_arel_max
    #     )
    
    
    # # else:
    # #     intrinsic = an['intrinsic']
    # #     cam_params = {'fx': intrinsic[0], 'fy': intrinsic[1], 'cx': intrinsic[2], 'cy': intrinsic[3], 'dataset': ''}
    # #     img_h, img_w = rgb_torch.shape[1], rgb_torch.shape[2]
    # #     erp_h = 1400
    # #     img_out, depth_out, valid_mask, active_mask, depth_out_gt = erp_patch_to_cam_fast(
    # #         rgb_torch.detach().cpu(), pred_depth.unsqueeze(0).detach().cpu(), torch.ones_like(gt_depth.unsqueeze(0)).to("cpu"), 0, 0, out_h=img_h, out_w=img_w, erp_h=erp_h, erp_w=erp_h*2, cam_params=cam_params, depth_erp_gt=gt_depth.unsqueeze(0).detach().cpu())
    # #     rgb = save_val_imgs_v2(
    # #         i,
    # #         depth_out,
    # #         depth_out_gt,
    # #         img_out,
    # #         f'rgb_{i:06d}_remap_org_cam.jpg',
    # #         save_dir=save_imgs_dir,
    # #         active_mask=active_mask,
    # #         valid_depth_mask=valid_mask,
    # #         depth_max=vis_depth_max,
    # #         arel_max=vis_arel_max
    # #     )
    # #save_raw_imgs(pred_depth.detach().cpu().numpy(), rgb_torch, osp.join(an['folder'], an['filename']), save_imgs_dir, 1000.0)

    # # pcd
    # pred_depth = pred_depth.detach().cpu().numpy()
    #pcd = reconstruct_pcd(pred_depth, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
    #os.makedirs(osp.join(save_pcd_dir, an['folder']), exist_ok=True)
    #save_point_cloud(pcd.reshape((-1, 3)), rgb_origin.reshape(-1, 3), osp.join(save_pcd_dir, an['folder'], an['filename'][:-4]+'.ply'))
    if main_process():
        eval_error = dam.get_metrics()
        print('>>>>>W/o scale: ', eval_error)
        
    if an['intrinsic'] == None:
        #for r in [0.9, 1.0, 1.1]:
        # for r in [1.0]:
            #for f in [600, 800, 1000, 1250, 1500]:
            # for f in [1000]:
                # pcd = reconstruct_pcd(pred_depth, f * r, f * (2-r), intrinsic[2], intrinsic[3])
        pcd = reconstruct_pcd_erp(pred_depth, mask)
        # fstr = '_fx_' + str(int(f * r)) + '_fy_' + str(int(f * (2-r)))
        os.makedirs(osp.join(save_pcd_dir, an['folder']), exist_ok=True)
        save_point_cloud(pcd.reshape((-1, 3)), rgb_origin.reshape(-1, 3), osp.join(save_pcd_dir, an['folder'], an['filename'][:-4] +'.ply'))
        
        # save gt pcd
        gt_depth = gt_depth.detach().cpu().numpy()
        # convert to float32
        gt_depth = gt_depth.astype(np.float32)
        gt_pcd = reconstruct_pcd_erp(gt_depth, mask)
        os.makedirs(osp.join(save_pcd_dir, an['folder'], 'gt'), exist_ok=True)
        save_point_cloud(gt_pcd.reshape((-1, 3)), rgb_origin.reshape(-1, 3), osp.join(save_pcd_dir, an['folder'], 'gt', an['filename'][:-4] +'.ply'))
        

    # if normal_out is not None:
    #     pred_normal = normal_out[:3, :, :] # (3, H, W)
    #     H, W = pred_normal.shape[1:]
    #     pred_normal = pred_normal[ :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

    #     gt_normal = None
    #     #if gt_normal_flag:
    #     if False:
    #         pred_normal = torch.nn.functional.interpolate(pred_normal, size=gt_normal.shape[2:], mode='bilinear', align_corners=True)    
    #         gt_normal = cv2.imread(norm_path)
    #         gt_normal = cv2.cvtColor(gt_normal, cv2.COLOR_BGR2RGB) 
    #         gt_normal = np.array(gt_normal).astype(np.uint8)
    #         gt_normal = ((gt_normal.astype(np.float32) / 255.0) * 2.0) - 1.0
    #         norm_valid_mask = (np.linalg.norm(gt_normal, axis=2, keepdims=True) > 0.5)
    #         gt_normal = gt_normal * norm_valid_mask               
    #         gt_normal_mask = ~torch.all(gt_normal == 0, dim=1, keepdim=True)
    #         dam.update_normal_metrics_gpu(pred_normal, gt_normal, gt_normal_mask, cfg.distributed)# save valiad normal

    #     save_normal_val_imgs(iter, 
    #                         pred_normal, 
    #                         gt_normal if gt_normal is not None else torch.ones_like(pred_normal, device=pred_normal.device),
    #                         rgb_torch, # data['input'], 
    #                         osp.join(an['folder'], 'normal_'+an['filename']), 
    #                         save_imgs_dir,
    #                         )

