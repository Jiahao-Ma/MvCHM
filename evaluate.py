import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

import argparse
from lib.utils.config_utils import cfg, cfg_from_yaml_file
from model.stem.mvchm import MvCHM
from lib.utils.tool_utils import gen_scale, to_numpy
from lib.utils.nms_utils import heatmap_nms
from lib.data.multiviewX import MultiviewX

from lib.data.wildtrack import Wildtrack
from lib.data.dataloader import get_padded_value, collater, MultiviewDataset
import random
from lib.evaluation.evaluate import evaluate_rcll_prec_moda_modp
from lib.utils.depth_utils import get_imagecoord_from_worldcoord
import matplotlib.pyplot as plt    
from lib.utils.visual_utils import Process
# matplotlib.use('TkAgg')              

def setup_seed(seed=7777):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def encode_postion(heatmap, mode, grid_reduce, thresh=None, nms=False, _mask=True, edge=20):
    if _mask:
        mask = torch.Tensor(np.load('n_mask.npy')).to(device=heatmap.device)
        mask = torch.where(mask < 1)
    assert mode in ['gt', 'pred']
    if len(heatmap.shape) != 2:
        heatmap=heatmap.squeeze(0).squeeze(-1)
    # heatmap_masks = torch.zeros_like(heatmap)
    # heatmap_masks[edge:-edge, edge:-edge] = 1
    # heatmap = heatmap * heatmap_masks
    # if len(offset_xy.shape) != 3:
    #     offset_xy = offset_xy.squeeze(0)
    if nms:
        heatmap = heatmap_nms(heatmap.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    if mode == 'pred':
        heatmap = torch.sigmoid(heatmap)
        if _mask:
            heatmap[mask] = 0
    heatmap = to_numpy(heatmap)
    # offset_xy = to_numpy(offset_xy)
    if mode == 'gt':
        xx, yy = np.where(heatmap == 1)
    elif mode == 'pred':
        assert thresh is not None 
        xx, yy = np.where(heatmap >= thresh)
    # offset_xy = offset_xy[xx, yy]
    # pos_x = (xx + offset_xy[:,0]) * grid_reduce
    # pos_y = (yy + offset_xy[:,1]) * grid_reduce
    pos_x = xx * grid_reduce
    pos_y = yy * grid_reduce
    pos = np.stack([pos_x, pos_y, np.zeros_like(pos_x)], axis=-1)
    return pos

class FormatPRData():
    def __init__(self, save_dir) -> None:
        self.data = None
        self.save_dir = save_dir

    def add_item(self, location, id):
        
        if self.data is None:
            self.data = np.concatenate([ np.ones((location.shape[0], 1))*id,  location], axis=1)
        else:
            tmp = np.concatenate([ np.ones((location.shape[0], 1))*id,  location], axis=1)
            self.data = np.concatenate([self.data, tmp], axis=0)
    def save(self):
        if not os.path.exists(os.path.dirname(self.save_dir)):
            os.mkdir(os.path.dirname(self.save_dir))
        np.savetxt(self.save_dir, self.data)
    
    def exist(self):
        return os.path.exists(self.save_dir)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=r'cfgs\MvDDE.yaml',\
         help='specify the config for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')  
    parser.add_argument('--cfg_file', type=str, default=None, help='the path to trained model checkpoint')  
    parser.add_argument('--dataname', type=str, default='Wildtrack', help='the name of dataset')

    parser.add_argument('--data_root', type=str, default=None, help='the path of dataset. eg: /path/to/Wildtrack')
    
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main(thresh=0.5):

    args, cfg = parse_config()
    setup_seed(0)
    # define preprocess operation and dataloader
    cfg_file = args.cfg_file
    # cfg_file = r'F:\ANU\ENGN8602\Code\MvDDE\MvCHM\experiments\2022-10-23_19-53-52_wt\MvDDE.yaml'
    cfg_from_yaml_file(cfg_file, cfg)

    new_w, new_h, old_w, old_h = cfg.DATA_CONFIG.NEW_WIDTH, cfg.DATA_CONFIG.NEW_HEIGHT, cfg.DATA_CONFIG.OLD_WIDTH, cfg.DATA_CONFIG.OLD_HEIGHT
    scale, scale_h, scale_w = gen_scale(new_w, new_h, old_w, old_h)
    pad_h, pad_w = get_padded_value(new_h, new_w)    

    process = Process(scale_h, scale_w, pad_h, pad_w, new_h, new_w, old_h, old_w)

    assert args.data_root is not None, 'Please specify the path of dataset'
    assert args.dataname in ['MultiviewX', 'Wildtrack'], 'Please specify the name of dataset'
    dataname = args.dataname
    path = args.data_root
    # dataname = 'Wildtrack'
    # path = 'F:\ANU\ENGN8602\Data\{}'.format(dataname) # MultiviewX
    DATASET = {'MultiviewX': MultiviewX, 'Wildtrack': Wildtrack}

    val_dataset = MultiviewDataset( DATASET[dataname](root=path), set_name='val')
    val_dataloader = DataLoader( val_dataset, num_workers=1, batch_size=1, collate_fn=collater)

    device = torch.device('cuda:0')

    # define model
    model = MvCHM(cfg, val_dataset, process, device)
    experiment_root = r"F:\ANU\ENGN8602\Code\MvDDE\MvCHM\experiments\2022-10-23_19-53-52_wt"
    chpt = "Epoch40_train_loss0.0608_val_loss0.0864.pth"
    # ck_file = os.path.join(experiment_root, "checkpoints", chpt)
    ck_file = r"checkpoints\Wildtrack.pth"

    model.load_state_dict(torch.load(ck_file, map_location=torch.device('cuda:0'))['model_state_dict'])
    model.to(device=device)
    model.eval()

    pr_dir_pred = os.path.join(experiment_root, chpt[:-4], "pr_dir_pred.txt")
    pr_dir_gt = os.path.join(experiment_root, chpt[:-4], "pr_dir_gt.txt")

    eval_tool = 'matlab'
    PR_pred = FormatPRData(pr_dir_pred)
    PR_gt = FormatPRData(pr_dir_gt)
    
    # grid 
    # xi = np.arange(0, 480, 1)
    # yi = np.arange(0, 1440, 1)
    # world_grid = np.stack(np.meshgrid(xi, yi, indexing='xy')).reshape([2, -1])

    # world_grid = world_grid.transpose()
    # idx =  np.where(np.logical_or(np.logical_or(world_grid[:, 0] == 0, world_grid[:, 0] == max(world_grid[:, 0])) , 
    #                 np.logical_or(world_grid[:, 1] == 0 ,world_grid[:, 1] == max(world_grid[:, 1]))))
    # world_grid = world_grid[idx]
    # world_grid = world_grid.transpose()

    # world_coord = Wildtrack.get_worldcoord_from_worldgrid(world_grid)

    if not PR_pred.exist() or not PR_gt.exist() or True:
        with tqdm(iterable=val_dataloader, desc=f'[EVALUATE] ', postfix=dict, mininterval=1) as pbar:
            for batch_idx, data in enumerate(val_dataloader):
                heatmap = torch.Tensor(data['heatmap']).to(device)
                with torch.no_grad():
                    batch_dict, batch_pred = model(data)
                gt_pos = encode_postion(heatmap = heatmap, mode = 'gt', grid_reduce = val_dataset.base.grid_reduce)
                pred_pos = encode_postion(heatmap = batch_pred['heatmap'], mode = 'pred', grid_reduce = val_dataset.base.grid_reduce, thresh = thresh, nms=True) # (n, 3)
                heatmap = batch_pred['heatmap'].squeeze(0).squeeze(-1).cpu().numpy()
                heatmap = (np.clip(heatmap, a_min=0, a_max=1.0) * 255).astype(np.uint8)
                # if not os.path.exists('visualization\\pred'):
                #     os.makedirs('visualization\\pred')
                # fig = plt.figure(figsize=(15, 15))
                # axes = fig.subplots(3,3).reshape(-1)
                # axes[0].set_title('gt')
                # axes[0].imshow(data['heatmap'])
                # axes[0].axis('off')
                # axes[1].set_title('pred')
                # axes[1].imshow(heatmap)
                # axes[1].axis('off')
                # # visualize the person idx on the BEV heatmap
                # for idx, pos in enumerate(pred_pos.copy()):
                #     pos = pos[:2] / val_dataset.base.grid_reduce
                #     axes[1].text(pos[1], pos[0]+1, f'{idx}', color='red')
                
                # pos_world_coord = Wildtrack.get_worldcoord_from_worldgrid(pred_pos.T[:2]) # (2, n)
                # person_ids_all = np.arange(pos_world_coord.shape[1])
                # for cam, path in enumerate(data['image_paths']):
                #     # visualize the grid in the image
                    # img_coord = get_imagecoord_from_worldcoord(world_coord, val_dataset.intrinsic_matrices[cam],
                #                                                         val_dataset.extrinsic_matrices[cam])
                #     img_coord = img_coord[:, np.where((img_coord[0] > 0) & (img_coord[1] > 0) &
                #                                     (img_coord[0] < 1920) & (img_coord[1] < 1080))[0]]
                #     axes[cam+2].scatter(img_coord.T[:, 0], img_coord.T[:, 1], color='green',s=1)        
                #     # visualize the person idx in the image
                #     img_coord_person = get_imagecoord_from_worldcoord(pos_world_coord, val_dataset.intrinsic_matrices[cam], val_dataset.extrinsic_matrices[cam])
                #     mask = np.where((img_coord_person[0] > 0) & (img_coord_person[1] > 0) & (img_coord_person[0] < 1920) & (img_coord_person[1] < 1080))[0]
                #     img_coord_person = img_coord_person.T
                #     person_ids = person_ids_all[mask]
                #     img_coord_person = img_coord_person[mask]
                #     for p_id, p_coord in zip(person_ids, img_coord_person):
                #         axes[cam+2].text(p_coord[0], p_coord[1], str(p_id), color='red')
                    
                #     # visualize image
                #     img = Image.open(path)
                #     axes[cam+2].imshow(img)
                #     axes[cam+2].axis('off')
                #     axes[cam+2].set_title(f'Cam{cam}')
                    
                #     # visualize keypoints
                #     kyp = batch_dict['kyps_align'][cam].squeeze(1)
                #     axes[cam+2].scatter(kyp[:, 0], kyp[:, 1], color='purple', s=1)
                #     # for bbox in batch_dict['pred_boxes'][cam]:
                #     #     x1,y1,x2,y2 = bbox[:4]
                #     #     w = x2 - x1
                #     #     h = y2 - y1
                #     #     rect = plt.Rectangle((x1, y1), w, h, fill=False, color='blue')
                #     #     axes[cam+2].add_patch(rect)
                    
                # axes = axes.reshape(3,3)
                # plt.savefig("visualization\\pred\\%d.jpg" % batch_idx, bbox_inches='tight',dpi=300, pad_inches=0.0 )
                # plt.close()
                print("gt: ", gt_pos.shape, " pred: ", pred_pos.shape)
                PR_pred.add_item(pred_pos, batch_idx)
                PR_gt.add_item(gt_pos, batch_idx)
                pbar.update(1)
        
        PR_pred.save()
        PR_gt.save()
    print(chpt)
    print('thresh: ', thresh)
    recall, precision, moda, modp = evaluate_rcll_prec_moda_modp(pr_dir_pred, pr_dir_gt, dataset=cfg.DATA_CONFIG.DATASET, eval=eval_tool)
    print(f'\n{eval_tool} eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')

if __name__ == '__main__':
    main(thresh=0.86)
    '''
    Wildtrack:
    thresh = 0.96:  MODA 94.0, MODP 83.3, prec 96.4, rcll 97.7
    thresh = 0.965: MODA 93.9, MODP 83.4, prec 96.3, rcll 97.7
    thresh = 0.97:  MODA 93.9, MODP 84.0, prec 97.0, rcll 97.0
    
    MultiviewX:
    Epoch29_train_loss0.0062_val_loss0.0038
    thresh = 0.90:  MODA 89.1, MODP 84.9, prec 97.2, rcll 91.7
    thresh = 0.88:  MODA 89.7, MODP 85.0, prec 96.5, rcll 93.0
    thresh = 0.86:  MODA 89.1, MODP 84.9, prec 95.2, rcll 93.8
    thresh = 0.85:  MODA 90.1, MODP 85.4, prec 95.5, rcll 94.5
    thresh = 0.84:  MODA 89.9, MODP 84.9, prec 95.1, rcll 94.7
    thresh = 0.83:  MODA 89.7, MODP 85.1, prec 94.8, rcll 94.8
    
    Epoch41_train_loss0.0011_val_loss0.0012.pth
    thresh = 0.84:  MODA 93.0, MODP 88.6, prec 97.3, rcll 95.6
    thresh = 0.85:  MODA 93.0, MODP 88.0, prec 97.5, rcll 95.4
    thresh = 0.88:  MODA 93.1, MODP 87.9, prec 98.2, rcll 94.8
    thresh = 0.89:  MODA 93.4, MODP 88.3, prec 98.5, rcll 94.8
    thresh = 0.90:  MODA 92.9, MODP 88.5, prec 98.4, rcll 94.4
    
    2022-10-22_16-27-12_wt
    Wildtrack:
    Epoch16_train_loss0.0157_val_loss0.3674.pth
    thresh = 0.8  MODA 82.7, MODP 81.5, prec 90.1, rcll 92.9
    thresh = 0.86 MODA 83.4, MODP 82.1, prec 92.0, rcll 91.3
    
    Epoch10_train_loss0.0241_val_loss0.1633.pth
    thresh = 0.86 MODA 80.9, MODP 81.2, prec 88.2, rcll 93.3
    
    Epoch31_train_loss0.0038_val_loss0.8403:
    thresh = 0.86 MODA 84.6, MODP 82.6, prec 94.0, rcll 90.4
    
    Epoch20_train_loss0.0105_val_loss0.4799
    thresh = 0.86 MODA 82.3, MODP 81.2, prec 91.4, rcll 90.8
    
    Epoch25_train_loss0.0064_val_loss0.6545.pth
    thresh = 0.86 MODA 85.1, MODP 82.7, prec 93.0, rcll 92.1
    
    Epoch29_train_loss0.0045_val_loss0.8878.pth
    thresh = 0.86 MODA 84.7, MODP 83.0, prec 93.7, rcll 90.8
    
    Epoch28_train_loss0.0048_val_loss0.7834.pth
    thresh = 0.86 MODA 84.1, MODP 82.3, prec 92.5, rcll 91.5
    
    2022-10-23_19-53-52_wt
    Epoch29_train_loss0.0640_val_loss0.0756.pth
    thresh = 0.86 
    '''