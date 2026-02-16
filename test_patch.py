import os
import argparse
from tqdm import tqdm
from config import cfg
from copy import deepcopy

parser = argparse.ArgumentParser(description='VIC test patch - per-frame density outputs')
parser.add_argument('--DATASET', type=str, default='MovingDroneCrowd')
parser.add_argument('--output_dir', type=str, default='test_results_patch')
parser.add_argument('--test_name', type=str, default='check')
parser.add_argument('--test_intervals', type=int, default=4)
parser.add_argument('--skip_flag', type=bool, default=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--GPU_ID', type=str, default='0')
opt = parser.parse_args()
opt.output_dir = os.path.join(opt.output_dir, opt.DATASET, opt.test_name)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID

import torch
import datasets
import torch.nn.functional as F
import numpy as np
import cv2
from model.VIC import Video_Counter
from misc.layer import Gaussianlayer
from misc.utils import change2map
from train import compute_metrics_all_scenes

def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        state_dict[k] = v
    return state_dict

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_map_images(img_tensor, pred_map, gt_map, restore_transform, out_dir, frame_idx):
    # ensure out_dir exists
    os.makedirs(out_dir, exist_ok=True)

    # img_tensor: torch tensor (C,H,W) (already padded)
    img_np = cv2.cvtColor(np.array(restore_transform(img_tensor.cpu())), cv2.COLOR_RGB2BGR)

    # get numpy arrays (allow both (H,W) and (1,H,W))
    pred_np = pred_map.detach().cpu().numpy()
    gt_np = gt_map.detach().cpu().numpy()
    if pred_np.ndim == 2:
        pred_np = np.expand_dims(pred_np, 0)
    if gt_np.ndim == 2:
        gt_np = np.expand_dims(gt_np, 0)

    pred_color = change2map(pred_np.copy())
    gt_color = change2map(gt_np.copy())

    # resize heatmaps to image size
    pred_resized = cv2.resize(pred_color, (img_np.shape[1], img_np.shape[0]))
    gt_resized   = cv2.resize(gt_color,   (img_np.shape[1], img_np.shape[0]))

    # overlays
    overlay_pred = cv2.addWeighted(img_np, 0.6, pred_resized, 0.4, 0)
    overlay_gt   = cv2.addWeighted(img_np, 0.6, gt_resized,   0.4, 0)
    both_maps    = cv2.addWeighted(pred_resized, 0.5, gt_resized, 0.5, 0)
    overlay_both = cv2.addWeighted(img_np, 0.6, both_maps, 0.4, 0)

    # save separate images
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_img.jpg"), img_np)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_pred_den.jpg"), pred_color)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_gt_den.jpg"), gt_color)

    # save overlays
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_overlay_pred.jpg"), overlay_pred)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_overlay_gt.jpg"), overlay_gt)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_overlay_both.jpg"), overlay_both)
    
def test(cfg_data):
    model = Video_Counter(cfg, cfg_data)
    Gaussian = Gaussianlayer()
    model.cuda()
    Gaussian.cuda()

    test_loader, restore_transform = datasets.loading_testset(opt.DATASET, opt.test_intervals, opt.skip_flag, mode='test')
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(module2model(state_dict), strict=True)
    model.eval()

    if opt.skip_flag:
        intervals = 1
    else:
        intervals = opt.test_intervals

    ensure_dir(opt.output_dir)

    for scene_id, (scene_name, sub_valset) in enumerate(test_loader, 0):
        scene_out = os.path.join(opt.output_dir, scene_name)
        ensure_dir(scene_out)
        gen_tqdm = tqdm(sub_valset, desc=scene_name)
        for vi, data in enumerate(gen_tqdm, 0):
            if vi % opt.test_intervals == 0 or vi == len(sub_valset) - 1:
                frame_signal = 'match'
            else:
                frame_signal = 'skip'

            if frame_signal == 'match' or not opt.skip_flag:
                img, label = data
                for i in range(len(label)):
                    for key, val in label[i].items():
                        if torch.is_tensor(val):
                            label[i][key] = val.cuda()
                img = img.cuda()
                with torch.no_grad():
                    b, c, h, w = img.shape
                    pad_h = 0 if h % 32 == 0 else (32 - h % 32)
                    pad_w = 0 if w % 32 == 0 else (32 - w % 32)
                    img = F.pad(img, (0, pad_w, 0, pad_h), "constant")
                    h, w = img.size(2), img.size(3)

                    pre_map, gt_den, *_ = model(img, label)
                    # make negatives zero (like original)
                    # (we only use global density for counting)
                    pred_cnt = float(pre_map[0].sum().item())
                    gt_cnt = float(gt_den[0].sum().item())

                    # save per-frame count CSV
                    counts_path = os.path.join(scene_out, "counts.csv")
                    header = False
                    if not os.path.exists(counts_path):
                        header = True
                    with open(counts_path, "a") as f:
                        if header:
                            f.write("frame,pred_count,gt_count\n")
                        f.write(f"{vi},{pred_cnt:.4f},{gt_cnt:.4f}\n")

                    # save separate images for this frame
                    img0 = img[0]
                    pred_map0 = pre_map[0]
                    gt_map0 = gt_den[0]
                    save_map_images(img0, pred_map0, gt_map0, restore_transform, scene_out, vi)

    print("Done. Results saved to:", opt.output_dir)

if __name__ == '__main__':
    import numpy as np
    import torch
    from importlib import import_module

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    datasetting = import_module(f'datasets.setting.{opt.DATASET}')
    cfg_data = datasetting.cfg_data
    test(cfg_data)