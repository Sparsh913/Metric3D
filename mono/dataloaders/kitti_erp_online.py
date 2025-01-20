import os
import json
import random
import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset
from mono.utils.erp_geometry import cam_to_erp_patch_fast


class KITTIERPOnlineDataset(BaseDataset):
    CAM_INTRINSIC = {
        "2011_09_26": np.array(
            [
                [7.215377e02, 0.000000e00, 6.095593e02, 4.485728e01],
                [0.000000e00, 7.215377e02, 1.728540e02, 2.163791e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.745884e-03],
            ]
        ).astype(np.float32),
        "2011_09_28": np.array(
            [
                [7.070493e02, 0.000000e00, 6.040814e02, 4.575831e01],
                [0.000000e00, 7.070493e02, 1.805066e02, -3.454157e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 4.981016e-03],
            ]
        ).astype(np.float32),
        "2011_09_29": np.array(
            [
                [7.183351e02, 0.000000e00, 6.003891e02, 4.450382e01],
                [0.000000e00, 7.183351e02, 1.815122e02, -5.951107e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.616315e-03],
            ]
        ).astype(np.float32),
        "2011_09_30": np.array(
            [
                [7.070912e02, 0.000000e00, 6.018873e02, 4.688783e01],
                [0.000000e00, 7.070912e02, 1.831104e02, 1.178601e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 6.203223e-03],
            ]
        ).astype(np.float32),
        "2011_10_03": np.array(
            [
                [7.188560e02, 0.000000e00, 6.071928e02, 4.538225e01],
                [0.000000e00, 7.188560e02, 1.852157e02, -1.130887e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 3.779761e-03],
            ]
        ).astype(np.float32),
    }
    min_depth = 0.01
    max_depth = 80
    test_split = "kitti_eigen_test.txt"
    train_split = "kitti_eigen_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        is_dense=False,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        crop_size=(500, 1000),
        erp_height=1400,
        theta_aug_deg=0,
        phi_aug_deg=0,
        roll_aug_deg=0,
        fov_align=True,
        visual_debug = False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        # self.crop = crop
        self.is_dense = is_dense
        # the size of the output ERP patch
        self.height = crop_size[0]
        self.width = crop_size[1]
        # the size of the full ERP image
        self.erp_height = erp_height
        self.theta_aug_deg = theta_aug_deg
        self.phi_aug_deg = phi_aug_deg
        self.roll_aug_deg = roll_aug_deg
        self.fov_align = fov_align
        self.visual_debug = visual_debug

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join('splits/kitti', self.split_file)) as f:
            for line in f:
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == "None" or not os.path.exists(
                        os.path.join(self.base_path, depth_map)
                    ):
                        self.invalid_depth_num += 1
                        continue
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map
                    )
                    img_info["annotation_filename_erp_range"] = os.path.join(
                        self.base_path, depth_map.replace('.png', '.json')
                    )
                    
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                self.dataset.append(img_info)

        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        image_path = os.path.join(
                            # self.base_path,
                            self.dataset[idx]["image_filename"],
                        )
        image = np.asarray(
            Image.open(
                image_path
            )
        ).astype(np.uint8)
        depth = None
        if not self.benchmark:
            depth = (
                np.asarray(
                    Image.open(
                        os.path.join(
                            # self.base_path,
                            self.dataset[idx]["annotation_filename_depth"],
                        )
                    )
                ).astype(np.float32)
                / self.depth_scale
            )
            
        # prepare kitti image camera intrinsics
        cam_intrinsics = self.CAM_INTRINSIC[image_path.split("/")[2]][:, :3]
        cam_params = {
            'dataset': 'kitti',
            'wFOV': np.arctan(1242 / 2 / cam_intrinsics[0, 0]) * 2,
            'hFOV': np.arctan(375 / 2 / cam_intrinsics[1, 1]) * 2,
            'width': 1242,
            'height': 375,
            'fx': cam_intrinsics[0, 0],
            'fy': cam_intrinsics[1, 1],
            'cx': cam_intrinsics[0, 2],
            'cy': cam_intrinsics[1, 2],
        }
        
        # convert depth from zbuffer to euclidean distance
        if depth is not None:
            x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
            depth = depth * np.sqrt((x - cam_intrinsics[0, 2])**2 + (y - cam_intrinsics[1, 2])**2 + cam_intrinsics[0, 0]**2) / cam_intrinsics[0, 0]
            depth = depth.astype(np.float32)
        
        # prepare the erp patch, and associated labels
        theta = np.deg2rad(random.uniform(-self.theta_aug_deg, self.theta_aug_deg)).astype(np.float32)
        phi = np.deg2rad(random.uniform(-self.phi_aug_deg, self.phi_aug_deg)).astype(np.float32)
        roll = np.deg2rad(random.uniform(-self.roll_aug_deg, self.roll_aug_deg)).astype(np.float32)
        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > self.min_depth
            
        fov = cam_params['hFOV']
        if not self.test_mode and self.fov_align:
            scale_fac =  fov / ((self.height / self.erp_height) * np.pi)* 1.2 # 1.2 is for including more black border
        else:
            scale_fac = 1.0    
            
        erp_rgb, erp_depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            self.height, self.width, self.erp_height, self.erp_height*2, cam_params, roll, scale_fac=scale_fac
        )
        lat_range = np.array([float(np.min(latitude)), float(np.max(latitude))])
        long_range = np.array([float(np.min(longitude)), float(np.max(longitude))])
        
        info = self.dataset[idx].copy()
        # ERP output patch camera intrinsics (focal length has no actual meaning, just a simulation)
        info["camera_intrinsics"] = torch.tensor(
            [
                [1 / np.tan(np.pi/self.erp_height), 0.000000e00, self.width/2],
                [0.000000e00, 1 / np.tan(np.pi/self.erp_height), self.height/2],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
        
        # Image augmentation. Should only include those compatible with ERP
        image, gts, info = self.transform(image=(erp_rgb * 255.).astype(np.uint8), gts={"depth": erp_depth, "attn_mask": erp_mask}, info=info)
        
        if self.visual_debug:
            # visualize image, gts[gt], gts[attn_mask]
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow((image.permute(1, 2, 0) - image.min()) / (image.max() - image.min()))
            plt.title("Image")
            plt.subplot(2, 2, 2)
            plt.imshow(gts["gt"].squeeze())
            plt.title("Ground Truth")
            plt.subplot(2, 2, 3)
            plt.imshow(gts["attn_mask"].squeeze())
            plt.title("Attn Mask")
            plt.subplot(2, 2, 4)
            plt.imshow(gts["mask"].squeeze())
            plt.title("Valid Depth Mask")
            plt.show()
        
        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], 
                    "lat_range": lat_range, "long_range": long_range, 
                    "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], 
                    "lat_range": lat_range, "long_range": long_range}

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, width_start = int(image.shape[0] - self.height), int(
            (image.shape[1] - self.width) / 2
        )
        height_end, width_end = height_start + self.height, width_start + self.width
        image = image[height_start:height_end, width_start:width_end]
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"]
            if depth is not None:
                height_start, width_start = int(depth.shape[0] - self.height), int(
                    (depth.shape[1] - self.width) / 2
                )
                height_end, width_end = (
                    height_start + self.height,
                    width_start + self.width,
                )
                depth = depth[height_start:height_end, width_start:width_end]
                mask = depth > self.min_depth
                if self.test_mode:
                    mask = np.logical_and(mask, depth < self.max_depth)
                    # mask = self.eval_mask(mask)
                mask = mask.astype(np.uint8)
                new_gts["gt"] = depth
                new_gts["mask"] = mask
        if "attn_mask" in gts:
            attn_mask = gts["attn_mask"]
            if attn_mask is not None:
                attn_mask = attn_mask[height_start:height_end, width_start:width_end]
                new_gts["attn_mask"] = attn_mask

        return image, new_gts, info
