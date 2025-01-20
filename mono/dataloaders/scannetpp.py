"""
The dataset will convert the fisheye image to Half ERP image testing.
This version does NOT consider camera orientation to generate ERP pitch.

In testing
- if the model is trained from perspective images, align the erp virtual focal length to the target focal length. (default)
- if the mode is trained from erp images, the Half ERP image will be directly uses for testing.

"""

import os
import numpy as np
import cv2
import torch
from PIL import Image
import json

from .dataset import BaseDataset, resize_for_input
from mono.utils.erp_geometry import fisheye_kb_to_erp
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class ScanNetPPDataset(BaseDataset):
    min_depth = 0.01
    max_depth = 40
    test_split = "scannetpp_tiny_test_easy.txt"
    train_split = "scannetpp_tiny_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=1000,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        erp=False, # indicate whether the dataset is treated as erp (originally erp dataset can be treated as perspective for evaluation matching virtual f to tgt_f)
        tgt_f = 519, # focal length of perspective training data
        cano_sz=(1400, 1400), # half erp size of erp training data
        fwd_sz = (900, 900),
        visual_debug=False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.erp = erp
        self.tgt_f = tgt_f
        self.cano_sz = cano_sz
        self.fwd_sz = fwd_sz
        self.visual_debug = visual_debug
        
        self.crop_width = int(cano_sz[0])
        # make height so that the crop has the same aspect ratio as the fwd_sz, so thar resizing to fwd_sz will add margin
        self.crop_height = int(cano_sz[0] * fwd_sz[0] / fwd_sz[1])

        # load annotations
        self.scene_dict = {}
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join('splits/scannetpp', self.split_file)) as f:
            # cnt = 0
            for line in f:
                # if cnt % 5 == 0:
                #     continue
                # cnt += 1
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == "None":
                        self.invalid_depth_num += 1
                        continue
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map
                    )
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                
                # build the dictionary as nerf-studio format
                path_items = img_name.strip().split("/")
                scene_id = path_items[1]
                if scene_id not in self.scene_dict:
                    scene_transform_file = os.path.join(self.base_path, path_items[0], scene_id, path_items[2], 'nerfstudio', "transforms.json")
                    scene_info = json.load(open(scene_transform_file))
                    fisheye_grid = np.load(os.path.join(self.base_path, path_items[0], scene_id, path_items[2], 'grid_fisheye.npy'))
                    scene_info['fisheye_grid'] = fisheye_grid
                    self.scene_dict[scene_id] = scene_info
                 
                # TODO: filter by pitch angle to verify if that's the issue 
                                
                # ERP output patch camera intrinsics (focal length has no actual meaning, just a simulation)
                img_info["camera_intrinsics"] = torch.tensor(
                    [
                        [1 / np.tan(np.pi/self.cano_sz[1]), 0.000000e00, self.cano_sz[1]/2],
                        [0.000000e00, 1 / np.tan(np.pi/self.cano_sz[0]), self.cano_sz[0]/2],
                        [0.000000e00, 0.000000e00, 1.000000e00],
                    ]
                )
                                
                # setup pred_scale_factor due to conversion to target focal length
                # if not self.erp:
                img_info["pred_scale_factor"] = (img_info["camera_intrinsics"][0, 0] + img_info["camera_intrinsics"][1, 1]).item() / 2 / self.tgt_f
                img_info["camera_intrinsics"][0, 0] /= img_info["pred_scale_factor"]
                img_info["camera_intrinsics"][1, 1] /= img_info["pred_scale_factor"]
                # else:
                #     img_info["pred_scale_factor"] = 1.0
                
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
        image = np.asarray(
            Image.open(self.dataset[idx]["image_filename"])
        )
        depth = (
            np.asarray(
                cv2.imread(self.dataset[idx]["annotation_filename_depth"], cv2.IMREAD_ANYDEPTH)
            ).astype(np.float32)
            / self.depth_scale
        )
        info = self.dataset[idx].copy()
        path_items = self.dataset[idx]["image_filename"].strip().split("/")
        scene_id = path_items[3]
        
        # convert depth from zbuffer to euclid (critical for fisheye dataset to use euclid depth)
        # Because the depth is rendered, should it be converted back to euclid with undistorted ray direction? It requires the ray lookup table for efficiency
        fisheye_grid = self.scene_dict[scene_id]['fisheye_grid']
        fisheye_grid_z = cv2.resize(fisheye_grid[:, :, 2], (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
        depth = depth / fisheye_grid_z
        depth = depth.astype(np.float32)
        
        # convert fisheye image to half erp image
        image, depth, active_mask = fisheye_kb_to_erp(image, self.scene_dict[scene_id], output_size=self.cano_sz, depth_map=depth)
        # crop out area does not change depth scale
        y_start = int((self.cano_sz[0] - self.crop_height) / 2)
        y_end = y_start + self.crop_height
        image = image[y_start:y_end, :, :]
        depth = depth[y_start:y_end, :]
        active_mask = active_mask[y_start:y_end, :]
        info["camera_intrinsics"] = self.dataset[idx]["camera_intrinsics"].clone()
        
        # height, width = image.shape[:2]
        # height_start, height_end = int(height/4), int(height - height/4)
        # image = image[height_start:height_end, :, :]
        # depth = depth[height_start:height_end, :]
        
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask = resize_for_input(image, depth, self.fwd_sz, info["camera_intrinsics"], [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=active_mask)

        info['f_align_factor'] = np.copy(info['pred_scale_factor'])
        info['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor
        info['pad'] = pad
        info['scene_id'] = scene_id
        info['theta'] = 0
        info['phi'] = 0
        info['roll'] = 0
        if not self.test_mode:
            depth /= info['pred_scale_factor']
    
        image, gts, info = self.transform(image=image, gts={"depth": depth, "attn_mask": (attn_mask>0).astype(np.float32)}, info=info)
        lat_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
        long_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
        
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
            plt.imshow(gts["mask"].squeeze().bool())
            plt.title("valid Mask")
            plt.subplot(2, 2, 4)
            plt.imshow(gts["attn_mask"].squeeze().bool())
            plt.title("Attn Mask")
            plt.show()
            
        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_range": lat_range, "long_range":long_range, "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_range": lat_range, "long_range":long_range}
        
    # def get_pointcloud_mask(self, shape):
    #     mask = np.zeros(shape)
    #     height_start, height_end = 45, self.height - 9
    #     width_start, width_end = 41, self.width - 39
    #     mask[height_start:height_end, width_start:width_end] = 1
    #     return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, height_end = 0, self.fwd_sz[0]
        width_start, width_end = 0, self.fwd_sz[1]
        image = image[height_start:height_end, width_start:width_end]
        # info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        # info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start

        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"][height_start:height_end, width_start:width_end]
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

    # def eval_mask(self, valid_mask):
    #     border_mask = np.zeros_like(valid_mask)
    #     border_mask[15:465, 20:620] = 1  # prepared center region
    #     return np.logical_and(valid_mask, border_mask)
