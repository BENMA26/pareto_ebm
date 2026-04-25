# initialize samples from buffer

# sample with specific model and algorithm

# evaluate model with classifier for attribute coherence

# evaluate model with precision score for attribute coherence and quality

# evaluate model with fid score for attribute coherence and quality

# TO DO
# load model parameters
# load attributes
# identify logits
# print informations
# identify classifier logits

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import os
import argparse
from tqdm import tqdm
import json
import pickle 

import random
import torch
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torch.nn.utils import clip_grad_norm
import pytorch_lightning as pl
import wandb
from lightning.pytorch.utilities import rank_zero_only

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model.model import classifier_lightning_module,ebm_lightning_module,jem_lightning_module,pareto_classifier_lightning_module,pareto_jem_lightning_module
from model.data import get_data,get_data_zero_shot,get_data_specific_subset
from model.utils import clamp_x, ReplayBuffer, ReservoirBuffer
from model.callbacks import SaveReplayBufferCallback
import matplotlib.pyplot as plt
from model.constant import CONFLICTS_LIST, ATTRIBUTE_LIST

from torchjd import backward,mtl_backward
from torchjd.aggregation import MGDA,UPGrad,DualProj,AlignedMTL

import argparse

import json
import os

# Default metadata paths (can be overridden via environment variables).
ATTR_NAMES_JSON = os.environ.get(
    "PARETO_EBM_ATTR_NAMES_JSON",
    str(PROJECT_ROOT / "artifacts" / "attr_names.json"),
)
THRESH_JSON = os.environ.get(
    "PARETO_EBM_THRESHOLD_JSON",
    str(PROJECT_ROOT / "artifacts" / "classifier_threshold.json"),
)
# --------- helpers ---------
import os
import numpy as np
import torch
from PIL import Image

def save_sample_list_as_jpg(sample_list, save_dir, start_idx):
    """
    Save a [0,1]-range image list (torch.Tensor or np.ndarray) as JPG files.
    Supported shapes: (H, W), (H, W, 1), (H, W, 3).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, img in enumerate(sample_list):
        # Step 1: convert to NumPy array
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        elif not isinstance(img, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(img)}")
        
        # Step 2: handle singleton/channel-first dimensions
        if img.ndim == 3:
            if img.shape[0] in (1, 3) and img.shape[1] > 3 and img.shape[2] > 3:
                # Likely CHW format (channel first).
                img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 2:
            pass  # Already HW.
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        # Step 3: clamp to [0, 1] and convert to uint8
        img = np.clip(img, 0.0, 1.0)
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Step 4: convert to PIL Image
        if img_uint8.ndim == 2:
            pil_img = Image.fromarray(img_uint8, mode='L')
        elif img_uint8.ndim == 3:
            if img_uint8.shape[2] == 1:
                pil_img = Image.fromarray(img_uint8.squeeze(), mode='L')
            elif img_uint8.shape[2] == 3:
                pil_img = Image.fromarray(img_uint8, mode='RGB')
            else:
                raise ValueError(f"Unsupported channel number: {img_uint8.shape[2]}")
        else:
            raise ValueError(f"Invalid image dimensions after processing: {img_uint8.shape}")
        
        # Step 5: save file
        filename = f"{start_idx + idx}.jpg"
        filepath = os.path.join(save_dir, filename)
        pil_img.save(filepath, quality=95)
    
    print(f"Saved {len(sample_list)} JPG images to '{save_dir}'.")

def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

_ATTR_CACHE = None
_THR_CACHE = None

def _load_attr_name_maps(attr_json_path: str = ATTR_NAMES_JSON):
    """
    Return:
      name_en_to_idx: dict
      name_zh_to_idx: dict
    Support formats:
      1) {"attributes":[{"index":0,"name_en":"Male","name_zh":"male_cn"}, ...]}
      2) [{"index":0,"name_en":"Male","name_zh":"male_cn"}, ...]
      3) ["Male", "Young", ...]  (index = position)
      4) {"name_en_list":[...]} or {"attr_names":[...]} etc.
    """
    global _ATTR_CACHE
    if _ATTR_CACHE is not None:
        return _ATTR_CACHE

    if not os.path.isfile(attr_json_path):
        raise FileNotFoundError(f"attr_names.json not found: {attr_json_path}")

    with open(attr_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # unwrap common containers
    if isinstance(obj, dict):
        if "attributes" in obj:
            obj = obj["attributes"]
        elif "name_en_list" in obj:
            obj = obj["name_en_list"]
        elif "attr_names" in obj:
            obj = obj["attr_names"]
        elif "names" in obj:
            obj = obj["names"]
        # else keep dict (we'll try parse below)

    name_en_to_idx, name_zh_to_idx = {}, {}

    # case A: list of dicts
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        for item in obj:
            idx = int(item.get("index"))
            ne = item.get("name_en", None)
            nz = item.get("name_zh", None)
            if ne is not None:
                name_en_to_idx[_norm(ne)] = idx
            if nz is not None:
                name_zh_to_idx[_norm(nz)] = idx

    # case B: list of strings (English names)
    elif isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], str)):
        for idx, ne in enumerate(obj):
            name_en_to_idx[_norm(ne)] = idx

    # case C: dict mapping
    elif isinstance(obj, dict):
        # e.g. {"Male":12, "Young":22} or {"12":"Male"} etc.
        for k, v in obj.items():
            # if value is int => k is name
            if isinstance(v, int):
                name_en_to_idx[_norm(k)] = int(v)
            # if key is int-like => value is name
            else:
                try:
                    idx = int(k)
                    if isinstance(v, str):
                        name_en_to_idx[_norm(v)] = idx
                except Exception:
                    pass
    else:
        raise ValueError(f"Unrecognized attr_names.json format: {type(obj)}")

    _ATTR_CACHE = (name_en_to_idx, name_zh_to_idx)
    return _ATTR_CACHE

def _load_threshold_list(threshold_json_path: str = THRESH_JSON):
    """
    Return: thresholds(list[float], len>=23)
    Support formats:
      1) {"best_thresholds":[...]}
      2) {"thresholds":[...]}
      3) {"classifier_threshold":[...]}
      4) [...]
    """
    global _THR_CACHE
    if _THR_CACHE is not None:
        return _THR_CACHE

    if not os.path.isfile(threshold_json_path):
        raise FileNotFoundError(f"classifier_threshold.json not found: {threshold_json_path}")

    with open(threshold_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        for key in ["best_thresholds", "thresholds", "classifier_threshold", "thr", "thresh"]:
            if key in obj:
                obj = obj[key]
                break

    if not isinstance(obj, list):
        raise ValueError(f"Unrecognized classifier_threshold.json format: {type(obj)}")

    thr = [float(x) for x in obj]
    _THR_CACHE = thr
    return thr

def get_index_from_attr_name(attr_name_1: str, attr_name_2: str,
                            attr_json_path: str = ATTR_NAMES_JSON):
    """
    Accept English/Chinese names (name_en/name_zh), case/underscore/space insensitive.
    """
    name_en_to_idx, name_zh_to_idx = _load_attr_name_maps(attr_json_path)

    def resolve(name: str) -> int:
        k = _norm(name)
        if k in name_en_to_idx:
            return name_en_to_idx[k]
        if k in name_zh_to_idx:
            return name_zh_to_idx[k]
        raise ValueError(
            f"Unknown attribute name '{name}'. "
            f"Not found in {attr_json_path}."
        )

    return resolve(attr_name_1), resolve(attr_name_2)

def get_threshold_from_index(index_1: int, index_2: int,
                             threshold_json_path: str = THRESH_JSON,
                             default: float = 0.5):
    """
    Read per-index thresholds from classifier_threshold.json.
    Fallback to `default` when index is out of range.
    """
    thr = _load_threshold_list(threshold_json_path)

    t1 = float(thr[index_1]) if 0 <= int(index_1) < len(thr) else float(default)
    t2 = float(thr[index_2]) if 0 <= int(index_2) < len(thr) else float(default)
    return t1, t2

def create_parser():
    parser = argparse.ArgumentParser(description='Process attributes and checkpoint directories.')

    parser.add_argument('--attr_name_1', type=str, required=True, 
                        help='First attribute name (string)')
    parser.add_argument('--attr_positive_1', action='store_true', 
                        help='Flag for first attribute being positive (store_true)')
    parser.add_argument('--attr_name_2', type=str, required=True, 
                        help='Second attribute name (string)')
    parser.add_argument('--attr_positive_2', action='store_true', 
                        help='Flag for second attribute being positive (store_true)')
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                        help='Checkpoint directory path (string)')
    parser.add_argument('--buffer_path', type=str, required=True, 
                        help='Buffer file path (string)')
    parser.add_argument('--save_path', type=str, 
                        help='save file path (string)')
    parser.add_argument('--model_name', type=str, 
                        help='save file path (string)')
    parser.add_argument('--num_rounds', type=int, 
                        help='save file path (string)')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='device')
    return parser

def sample(model1, model2, initial_samples, num_steps, learning_rate, sigma, device, retain_graph=False, return_samples_each_step=False):

        initial_samples = initial_samples.clone().detach().to(device)
        initial_samples.requires_grad_(True)   # Correct usage.

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)

            # compute energy
            energy_1 = model1(initial_samples).sum()
            energy_2 = model2(initial_samples).sum()
            # compute grad w.r.t. samples
            grad_1 = torch.autograd.grad(energy_1, initial_samples, create_graph=retain_graph)[0]
            grad_2 = torch.autograd.grad(energy_2, initial_samples, create_graph=retain_graph)[0]

            grad = (grad_1 + grad_2)
            '''
            loss_list = [energy_1,energy_2]
            aggregator = UPGrad()
            backward(
            tensors=loss_list,
            aggregator=aggregator,
            retain_graph=True,
            parallel_chunk_size=1,
            )
            grad = initial_samples.grad
            '''
            # Langevin update (do not use .data)
            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()  # detach old graph
            initial_samples.requires_grad_(True)  # re-enable grad

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

def main(cmd_args):
    args = argparse.Namespace(
        use_ema=True,
        exp_name=None,
        gpus=1,
        strategy='ddp',
        check_point_path=None,
        buffer_path=None,
        save_dir=os.getcwd(),
        dataset='celeba',
        celeba_drop_infreq=0.13,
        img_sigma=0.01,
        dataset_transform=False,
        label_smoothing=False,
        mix_up=False,
        epoch_num=10000,
        batch_size=128,
        num_workers=4,
        val_interval=1000,
        lr=2e-4,
        beta1=0.0,
        beta2=0.9,
        clip_gradients=False,
        gradient_clip_val=0.5,
        kl_loss=False,
        buffer_size=10000,
        replace_prob=0.05,
        buffer_transform=False,  # In train scripts this is action='store_true' (default False).
        sgld_steps=60,
        sgld_lr=100.0,
        sgld_sigma=0.001,
        clamp_grad=False,
        deep=True,
        multiscale=True,
        self_attn=True,
        filter_dim=64,
        img_size=64
    )

    classifier_check_point_path = f"{cmd_args.ckpt_dir}/classifier.ckpt"
    ckpt_path = f"{cmd_args.ckpt_dir}/{cmd_args.model_name}.ckpt"
    buffer_path = cmd_args.buffer_path

    index_1,index_2 = get_index_from_attr_name(cmd_args.attr_name_1,cmd_args.attr_name_2)

    threshold_1,threshold_2 = get_threshold_from_index(index_1,index_2)

    energy_net = pareto_jem_lightning_module(args).to(cmd_args.device)

    energy_net.load_state_dict(torch.load(ckpt_path)['state_dict'])

    with open(buffer_path,"rb") as file:
        buffer = pickle.load(file)

    for i in range(cmd_args.num_rounds):

        #initial_samples = buffer.sample(500)
        initial_samples = torch.rand(500,3,64,64)
        sample_list = energy_net.two_stage_latent_jem_pareto_multi_conditional_sample([index_1,index_2],[True if cmd_args.attr_positive_1 else False,True if cmd_args.attr_positive_2 else False],initial_samples,60,1,1e-3,return_samples_each_step=False)
        
        if cmd_args.model_name and cmd_args.save_path:
            save_dir = os.path.join(cmd_args.save_path,f"{cmd_args.model_name}_{cmd_args.attr_name_1}_{cmd_args.attr_positive_1}_{cmd_args.attr_name_2}_{cmd_args.attr_positive_2}")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_sample_list_as_jpg(sample_list, save_dir, i*500)

if __name__ == "__main__":
    parser = create_parser()
    cmd_args = parser.parse_args()
    print("computing!")
    print(f"{cmd_args.attr_name_1}_{cmd_args.attr_positive_1}")
    print(f"{cmd_args.attr_name_2}_{cmd_args.attr_positive_2}")
    main(cmd_args)
