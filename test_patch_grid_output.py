import os
import argparse
from tqdm import tqdm
from config import cfg
from copy import deepcopy

parser = argparse.ArgumentParser(description='VIC test patch - per-frame masked image output')
parser.add_argument('--DATASET', type=str, default='MovingDroneCrowd')
parser.add_argument('--output_dir', type=str, default='test_results_patch_grid')
parser.add_argument('--test_name', type=str, default='mask')
parser.add_argument('--test_intervals', type=int, default=4)
parser.add_argument('--skip_flag', type=bool, default=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--GPU_ID', type=str, default='0')
parser.add_argument('--grid_rows', type=int, default=8)
parser.add_argument('--grid_cols', type=int, default=8)
parser.add_argument('--dense_thr', type=float, default=2.0, help='density sum threshold per cell to consider it dense')
parser.add_argument('--mask_mode', type=str, default='blur', choices=['blur', 'mean', 'black'], help='how to mask dense cells')
parser.add_argument('--scene', type=str, default=None, help='If set, process only this scene (folder name)')
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

def grid_counts_from_density(density, rows, cols):
    if isinstance(density, torch.Tensor):
        dens = density.detach().cpu().numpy()
    else:
        dens = np.array(density)
    if dens.ndim == 3:
        dens = dens.squeeze(0)
    H, W = dens.shape
    ch = int(np.ceil(H / rows))
    cw = int(np.ceil(W / cols))
    counts = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        y0 = r * ch
        y1 = min((r + 1) * ch, H)
        for c in range(cols):
            x0 = c * cw
            x1 = min((c + 1) * cw, W)
            counts[r, c] = float(dens[y0:y1, x0:x1].sum())
    return counts

def mask_image_by_cells(img_np, dense_mask_cells, rows, cols, density_shape, mode='blur'):
    out = img_np.copy()
    den_h, den_w = density_shape
    img_h, img_w = img_np.shape[:2]
    ch = int(np.ceil(den_h / rows))
    cw = int(np.ceil(den_w / cols))
    for r in range(rows):
        for c in range(cols):
            if not dense_mask_cells[r, c]:
                continue
            y0 = r * ch
            y1 = min((r + 1) * ch, den_h)
            x0 = c * cw
            x1 = min((c + 1) * cw, den_w)
            y0_img = int(y0 * img_h / den_h)
            y1_img = int(y1 * img_h / den_h)
            x0_img = int(x0 * img_w / den_w)
            x1_img = int(x1 * img_w / den_w)
            if y1_img <= y0_img or x1_img <= x0_img:
                continue
            patch = out[y0_img:y1_img, x0_img:x1_img]
            if patch.size == 0:
                continue
            if mode == 'blur':
                kh = max(3, ((y1_img - y0_img) // 10) | 1)
                kw = max(3, ((x1_img - x0_img) // 10) | 1)
                k_h = kh if kh % 2 == 1 else kh + 1
                k_w = kw if kw % 2 == 1 else kw + 1
                try:
                    blurred = cv2.GaussianBlur(patch, (k_w, k_h), 0)
                except:
                    blurred = cv2.blur(patch, (5,5))
                out[y0_img:y1_img, x0_img:x1_img] = blurred
            elif mode == 'black':
                out[y0_img:y1_img, x0_img:x1_img] = (0, 0, 0)
            else:
                mean_color = [int(x) for x in cv2.mean(patch)[:3]]
                out[y0_img:y1_img, x0_img:x1_img] = mean_color
    return out

def save_masked_image(img_tensor, pre_map, restore_transform, out_dir, frame_idx, rows, cols, dense_thr, mask_mode):
    ensure_dir(out_dir)
    img_np = cv2.cvtColor(np.array(restore_transform(img_tensor.cpu())), cv2.COLOR_RGB2BGR)

    pred_np = pre_map.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_den = pred_np.squeeze(0)
    else:
        pred_den = pred_np

    pred_counts = grid_counts_from_density(pred_den, rows, cols)
    dense_mask_cells = pred_counts >= dense_thr

    masked_img = mask_image_by_cells(img_np, dense_mask_cells, rows, cols, pred_den.shape, mode=mask_mode)

    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_masked.jpg"), masked_img)

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
        if opt.scene is not None and scene_name != opt.scene:
            continue
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

                    # write per-frame summary counts (optional)
                    pred_cnt = float(pre_map[0].sum().item())
                    gt_cnt = float(gt_den[0].sum().item())
                    counts_path = os.path.join(scene_out, "counts.csv")
                    header = False
                    if not os.path.exists(counts_path):
                        header = True
                    with open(counts_path, "a") as f:
                        if header:
                            f.write("frame,pred_count,gt_count\n")
                        f.write(f"{vi},{pred_cnt:.4f},{gt_cnt:.4f}\n")

                    img0 = img[0]
                    pred_map0 = pre_map[0]

                    save_masked_image(img0, pred_map0, restore_transform, scene_out, vi, opt.grid_rows, opt.grid_cols, opt.dense_thr, opt.mask_mode)

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