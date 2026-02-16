import os
import argparse
from tqdm import tqdm
from config import cfg
from copy import deepcopy

parser = argparse.ArgumentParser(description='VIC test patch - per-frame grid counting')
parser.add_argument('--DATASET', type=str, default='MovingDroneCrowd')
parser.add_argument('--output_dir', type=str, default='test_results_patch_grid')
parser.add_argument('--test_name', type=str, default='check')
parser.add_argument('--test_intervals', type=int, default=4)
parser.add_argument('--skip_flag', type=bool, default=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--GPU_ID', type=str, default='0')
parser.add_argument('--grid_rows', type=int, default=8)
parser.add_argument('--grid_cols', type=int, default=8)
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
    # density: torch tensor (1,H,W) or numpy (H,W) or (1,H,W)
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

def draw_grid_overlay(img_bgr, counts, color_map=None, text_color=(255,255,255)):
    # img_bgr: numpy BGR image
    H, W = img_bgr.shape[:2]
    rows, cols = counts.shape
    ch = int(np.ceil(H / rows))
    cw = int(np.ceil(W / cols))
    out = img_bgr.copy()
    # draw semi-transparent cells with no intensity map (we assume heatmap is pre-blended)
    for r in range(rows):
        for c in range(cols):
            y0 = r * ch
            y1 = min((r + 1) * ch, H)
            x0 = c * cw
            x1 = min((c + 1) * cw, W)
            cv2.rectangle(out, (x0, y0), (x1, y1), (200,200,200), 1)
            val = counts[r, c]
            txt = f"{val:.1f}"
            # put text at 60% height of the cell
            tx = x0 + 3
            ty = y0 + int((y1-y0) * 0.6)
            scale = max(0.4, min(1.2, (cw/100.0)))
            cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)
    # also put total in top-left
    total = counts.sum()
    cv2.putText(out, f"Total: {total:.1f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, f"Total: {total:.1f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return out

def save_frame_grid_outputs(img_tensor, pred_map, gt_map, restore_transform, out_dir, frame_idx, rows, cols):
    ensure_dir(out_dir)
    # restore image to numpy BGR
    img_np = cv2.cvtColor(np.array(restore_transform(img_tensor.cpu())), cv2.COLOR_RGB2BGR)

    # ensure maps are shape (1,H,W) for change2map
    pred_np = pred_map.detach().cpu().numpy()
    gt_np = gt_map.detach().cpu().numpy()
    if pred_np.ndim == 2:
        pred_np = np.expand_dims(pred_np, 0)
    if gt_np.ndim == 2:
        gt_np = np.expand_dims(gt_np, 0)

    # heatmap images (colorized)
    pred_color = change2map(pred_np.copy())
    gt_color = change2map(gt_np.copy())

    # resize heatmaps to image size
    pred_resized = cv2.resize(pred_color, (img_np.shape[1], img_np.shape[0]))
    gt_resized = cv2.resize(gt_color, (img_np.shape[1], img_np.shape[0]))

    # produce overlays (image blended with heatmap)
    overlay_pred = cv2.addWeighted(img_np, 0.6, pred_resized, 0.4, 0)
    overlay_gt   = cv2.addWeighted(img_np, 0.6, gt_resized,   0.4, 0)

    # compute grid counts (on the map arrays at their native size)
    pred_counts = grid_counts_from_density(pred_map, rows, cols)
    gt_counts = grid_counts_from_density(gt_map, rows, cols)

    # draw grid + numbers on top of overlay images
    overlay_pred_grid = draw_grid_overlay(overlay_pred, pred_counts)
    overlay_gt_grid = draw_grid_overlay(overlay_gt, gt_counts)

    # save images
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_img.jpg"), img_np)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_pred_heat.jpg"), pred_resized)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_gt_heat.jpg"), gt_resized)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_overlay_pred_grid.jpg"), overlay_pred_grid)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_overlay_gt_grid.jpg"), overlay_gt_grid)

    # save counts CSV (one row per frame)
    csv_path = os.path.join(out_dir, "counts_grid.csv")
    header = False
    if not os.path.exists(csv_path):
        header = True
    flat_pred = pred_counts.flatten()
    flat_gt = gt_counts.flatten()
    total_pred = float(flat_pred.sum())
    total_gt = float(flat_gt.sum())
    with open(csv_path, "a") as f:
        if header:
            # frame,total_pred,total_gt, then per-cell columns
            cols = [f"cell_{r}_{c}" for r in range(rows) for c in range(cols)]
            f.write("frame,total_pred,total_gt," + ",".join(cols) + "\n")
        row = [str(frame_idx), f"{total_pred:.4f}", f"{total_gt:.4f}"] + [f"{v:.4f}" for v in flat_pred]
        f.write(",".join(row) + "\n")

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
                    pred_cnt = float(pre_map[0].sum().item())
                    gt_cnt = float(gt_den[0].sum().item())

                    # save per-frame numbers
                    counts_path = os.path.join(scene_out, "counts.csv")
                    header = False
                    if not os.path.exists(counts_path):
                        header = True
                    with open(counts_path, "a") as f:
                        if header:
                            f.write("frame,pred_count,gt_count\n")
                        f.write(f"{vi},{pred_cnt:.4f},{gt_cnt:.4f}\n")

                    # save grid visualizations and grid counts CSV
                    img0 = img[0]
                    pred_map0 = pre_map[0]
                    gt_map0 = gt_den[0]
                    save_frame_grid_outputs(img0, pred_map0, gt_map0, restore_transform, scene_out, vi, opt.grid_rows, opt.grid_cols)

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