
import os
import sys
import io
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ROS
from sensor_msgs.msg import Image
# import tf

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional

# models
# from vint_navid_vint_train.models.gnm.gnm import GNM
# from vint_train.models.vint.vint import ViNT

# from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import DenseNetwork
from vint_train.models.nomad.nomad import NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO

def pil_to_numpy_array(image_input, target_size: tuple = (224, 224)) -> np.ndarray:
    """Convert PIL image or numpy array to numpy array with proper formatting for Crossformer."""

    if isinstance(image_input, PILImage.Image):

        if image_input.size != target_size:
            image_input = image_input.resize(target_size)
        img_array = np.array(image_input)
    elif isinstance(image_input, np.ndarray):

        img_array = image_input.copy()

        if img_array.shape[:2] != target_size:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))
            elif len(img_array.shape) == 2:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8), mode='L')
            else:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))

            pil_temp = pil_temp.resize(target_size)
            img_array = np.array(pil_temp)
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    return img_array



def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    model_type = config["model_type"]
    
    if model_type == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif model_type == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)  
    if img.ndim > 3:  
        img = img[0] # removing batch size
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes() 
    ros_image.step = ros_image.width
    return ros_image
    
def pil_to_numpy_array(image_input, target_size: tuple = (224, 224)) -> np.ndarray:
    """Convert PIL image or numpy array to numpy array with proper formatting for Crossformer."""

    if isinstance(image_input, PILImage.Image):

        if image_input.size != target_size:
            image_input = image_input.resize(target_size)
        img_array = np.array(image_input)
    elif isinstance(image_input, np.ndarray):

        img_array = image_input.copy()

        if img_array.shape[:2] != target_size:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))
            elif len(img_array.shape) == 2:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8), mode='L')
            else:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))

            pil_temp = pil_temp.resize(target_size)
            img_array = np.array(pil_temp)
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    return img_array

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        if not isinstance(pil_img, PILImage.Image):
            pil_img = PILImage.fromarray(pil_img.astype(np.uint8))
        tensor_img = transforms.ToTensor()(pil_img)

        # Add fake channel if only 2 channels
        if tensor_img.shape[0] == 2:
            print("Adding fake third channel")
            fake_channel = torch.zeros_like(tensor_img[0:1])
            tensor_img = torch.cat([tensor_img, fake_channel], dim=0)

        pil_img = transforms.ToPILImage()(tensor_img)
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def rotate_point_by_quaternion(point, quaternion):
    r = R.from_quat(quaternion)
    rotation_matrix = r.as_matrix()

    point_h = np.array([point[0], point[1], point[2]])

    rotated_point = np.dot(np.linalg.inv(rotation_matrix), point_h)

    return rotated_point
