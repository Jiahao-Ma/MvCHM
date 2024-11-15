import os
import time
import torch
import shutil
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from model.stem.mvchm import MvCHM
from torch.utils.data import DataLoader
from lib.utils.visual_utils import Process
from lib.data.multiviewX import MultiviewX
from lib.data.wildtrack import Wildtrack
from lib.utils.nms_utils import heatmap_nms
from lib.utils.tool_utils import gen_scale, to_numpy
from lib.utils.config_utils import cfg, cfg_from_yaml_file
from lib.utils.depth_utils import get_imagecoord_from_worldcoord
from lib.data.dataloader import get_padded_value, collater, MultiviewDataset

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=r'cfgs\MvDDE.yaml',\
         help='specify the config for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')  
    
    parser.add_argument('--dataname', type=str, default='Wildtrack', help='the name of dataset')

    parser.add_argument('--data_root', type=str, default=None, help='the path of dataset. eg: /path/to/Wildtrack')
    
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def setup_seed(seed=7777):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def encode_postion(heatmap, mode, grid_reduce, thresh=None, nms=False, _mask=False):
    if _mask:
        mask = torch.Tensor(np.load('mask.npy')).to(device=heatmap.device)
        mask = torch.where(mask < 1)
    assert mode in ['gt', 'pred']
    if len(heatmap.shape) != 2:
        heatmap=heatmap.squeeze(0).squeeze(-1)

    if nms:
        heatmap = heatmap_nms(heatmap.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    if mode == 'pred':
        heatmap = torch.sigmoid(heatmap)
        if _mask:
            heatmap[mask] = 0
    heatmap = to_numpy(heatmap)

    if mode == 'gt':
        xx, yy = np.where(heatmap == 1)
    elif mode == 'pred':
        assert thresh is not None 
        xx, yy = np.where(heatmap >= thresh)

    pos_x = xx * grid_reduce
    pos_y = yy * grid_reduce
    pos = np.stack([pos_x, pos_y, np.zeros_like(pos_x)], axis=-1)
    return pos

def save_fig(data, title, path, keypoints=None, text=None, grid=None, text_size=23):
    ax = plt.figure(figsize=(15, 8)).add_subplot(111)
    ax.set_title(title)
    ax.imshow(data)
    data = np.array(data)
    if len(data.shape) == 2:
        H, W = data.shape
    else:
        H, W, _ = data.shape
    if keypoints is not None:
        for kyp in keypoints:  
            if kyp[0] > 0 and kyp[0] < W and kyp[1] > 0 and kyp[1] < H:
                ax.scatter(kyp[0], kyp[1], color='blue', s=4)
    if grid is not None:
        ax.scatter(grid[:, 0], grid[:, 1], color='purple', s=1)
    # for the figures of camera view
    if H == 1080 and W == 1920:
        H -= 20
        W -= 20 
    if text is not None:
        for t in text:
            if (t[2] < W) and (t[1] < H):
                ax.text(t[2], t[1]+1, f'{t[0]}', color='red', fontdict={'size':text_size})
    ax.axis('off')
    plt.savefig(path, bbox_inches='tight',dpi=300, pad_inches=0.0 )
    plt.close()

def main(thresh=0.7, save_idx=None):
    
    args, cfg = parse_config()
    setup_seed(0)
    # define preprocess operation and dataloader
    
    cfg_file = r'cfgs\MvDDE.yaml'
    cfg_from_yaml_file(cfg_file, cfg)

    new_w, new_h, old_w, old_h = cfg.DATA_CONFIG.NEW_WIDTH, cfg.DATA_CONFIG.NEW_HEIGHT, cfg.DATA_CONFIG.OLD_WIDTH, cfg.DATA_CONFIG.OLD_HEIGHT
    scale, scale_h, scale_w = gen_scale(new_w, new_h, old_w, old_h)
    pad_h, pad_w = get_padded_value(new_h, new_w)    

    process = Process(scale_h, scale_w, pad_h, pad_w, new_h, new_w, old_h, old_w)

    # dataname = 'Wildtrack'
    # path = 'F:\ANU\ENGN8602\Data\{}'.format(dataname) # MultiviewX
    assert args.data_root is not None, 'Please specify the path of dataset'
    assert args.dataname in ['MultiviewX', 'Wildtrack'], 'Please specify the name of dataset'
    dataname = args.dataname
    path = args.data_root
    DATASET = {'MultiviewX': MultiviewX, 'Wildtrack': Wildtrack}
    t0 = time.time()
    val_dataset = MultiviewDataset( DATASET[dataname](root=path), set_name='val')
    val_dataloader = DataLoader( val_dataset, num_workers=1, batch_size=1, collate_fn=collater)
    t1 = time.time()
    print(f'Dataloader time: {t1 - t0:.2f}s' )
    device = torch.device('cuda:0')

    # define model
    model = MvCHM(cfg, val_dataset, process, device)
 
    ck_file = r"checkpoints\Wildtrack.pth"

    model.load_state_dict(torch.load(ck_file, map_location=torch.device('cuda:0'))['model_state_dict'])
    model.to(device=device)
    model.eval()
    
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
    
    for batch_idx, data in enumerate(val_dataloader):
        print(f' Process {batch_idx} ...')
        if save_idx is not None:
            if batch_idx != save_idx:
                continue
        t2 = time.time()
        with torch.no_grad():
            batch_dict, batch_pred = model(data, save_pc=True)
        t3 = time.time()
        print(f'Inference time: {t3 - t2:.2f}s' )
        # gt_pos = encode_postion(heatmap = heatmap, mode = 'gt', grid_reduce = val_dataset.base.grid_reduce)
        pred_pos = encode_postion(heatmap = batch_pred['heatmap'], mode = 'pred', grid_reduce = val_dataset.base.grid_reduce, thresh = thresh, nms=True) # (n, 3)
        heatmap = batch_pred['heatmap'].squeeze(0).squeeze(-1).cpu().numpy()
        heatmap = (np.clip(heatmap, a_min=0, a_max=1.0) * 255).astype(np.uint8)
        
        # img_id = 0
        # save_path = r'visualization\appendix\{:02d}'.format(batch_idx)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # save_fig(data['heatmap'], 'gt', os.path.join(save_path, '{}.jpg'.format(img_id)))
        # # visualize the person idx on the BEV heatmap
        # pos_text = list()
        # for idx, pos in enumerate(pred_pos.copy()):
        #     pos = pos[:2] / val_dataset.base.grid_reduce
        #     pos_text.append([idx, *pos])
            
        # img_id+=1
        # # save_fig(heatmap, 'pred', os.path.join(save_path, '{}.jpg'.format(img_id)), text=pos_text)
        
        # pos_world_coord = Wildtrack.get_worldcoord_from_worldgrid(pred_pos.T[:2]) # (2, n)
        # person_ids_all = np.arange(pos_world_coord.shape[1])
        # for cam, path in enumerate(data['image_paths']):
        #     # visualize the grid in the image
        #     img_coord = get_imagecoord_from_worldcoord(world_coord, val_dataset.intrinsic_matrices[cam],
        #                                                         val_dataset.extrinsic_matrices[cam])
        #     img_coord = img_coord[:, np.where((img_coord[0] > 1) & (img_coord[1] > 1) &
        #                                     (img_coord[0] < 1918) & (img_coord[1] < 1078))[0]]
        #     # axes[cam+2].scatter(img_coord.T[:, 0], img_coord.T[:, 1], color='green',s=1)        
        #     # visualize the person idx in the image
        #     img_coord_person = get_imagecoord_from_worldcoord(pos_world_coord, val_dataset.intrinsic_matrices[cam], val_dataset.extrinsic_matrices[cam])
        #     mask = np.where((img_coord_person[0] > 0) & (img_coord_person[1] > 0) & (img_coord_person[0] < 1920) & (img_coord_person[1] < 1080))[0]
        #     img_coord_person = img_coord_person.T
        #     person_ids = person_ids_all[mask]
        #     img_coord_person = img_coord_person[mask]
        #     pos_text = list()
        #     for p_id, p_coord in zip(person_ids, img_coord_person):
        #         # axes[cam+2].text(p_coord[0], p_coord[1], str(p_id), color='red')
        #         pos_text.append([p_id, *p_coord[::-1]])
            
        #     # visualize image
        #     img = Image.open(path)
        #     # visualize keypoints
        #     kyp = batch_dict['kyps_align'][cam].squeeze(1)
        #     img_size = img.size
        #     mask = np.logical_and(np.logical_and(kyp[:,0]>0 , kyp[:,0] < (img_size[0]-2)), \
        #                           np.logical_and(kyp[:,1]>0 , kyp[:,1] < (img_size[0]-2)))
        #     kyp = kyp[mask]
        #     # for bbox in batch_dict['pred_boxes'][cam]:
        #     #     x1,y1,x2,y2 = bbox[:4]
        #     #     w = x2 - x1
        #     #     h = y2 - y1
        #     #     rect = plt.Rectangle((x1, y1), w, h, fill=False, color='blue')
        #     #     axes[cam+2].add_patch(rect)
            
        #     img_id+=1
        #     pos_text=None
        #     save_fig(img, f'Cam{cam}', os.path.join(save_path, '{}.jpg'.format(img_id)), keypoints=kyp, text=pos_text, grid=img_coord.T[:, :2], text_size=18)
            
            # plt.savefig(save_path, bbox_inches='tight',dpi=300, pad_inches=0.0 )
        # plt.savefig("visualization\\appendix\\%d.jpg" % batch_idx, bbox_inches='tight',dpi=300, pad_inches=0.0 )
        # plt.close()
    
def cat_figures():
    import cv2
    # delete heatmap
    for i in range(40):
        try:
            os.remove(r'visualization\appendix\{:02d}\9.jpg'.format(i))
            print('delete visualization\appendix\{:02d}\9.jpg'.format(i))
        except:
            pass
    for idx in os.listdir(r'visualization\appendix'):
        paths = [r'visualization\appendix\{}\{}.jpg'.format(idx, i) for i in range(2, 9)]
        imgs = [cv2.imread(p) for p in paths]
        # imgs = [cv2.cvtColor(img) for img in imgs]
        imgs = [cv2.resize(img, (480, 270)) for img in imgs]
        first_row = np.concatenate(imgs[:4], axis=1)
        second_row = np.concatenate(imgs[4:], axis=1)
        paths = [r'visualization\appendix\{}\{}.jpg'.format(idx, i) for i in range(0, 2)]
        imgs = [cv2.imread(p) for p in paths]
        # imgs = [cv2.cvtColor(img) for img in imgs]
        imgs = [cv2.resize(img, (480, 135)) for img in imgs]
        cv2.putText(imgs[0], 'GT', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(imgs[1], 'PRED', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        heatmaps = np.concatenate(imgs, axis=0)
        second_row = np.concatenate([second_row, heatmaps], axis=1)
        figure = np.concatenate([first_row, second_row], axis=0)
        
        cv2.imwrite(r'visualization\appendix\{}\9.jpg'.format(idx), figure)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 


def imgs2gif(imgs, saveName, duration=None, loop=0, fps=None):
    if fps:
        duration = 1 / fps
    duration *= 1000
    imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(saveName, save_all=True, append_images=imgs, duration=duration, loop=loop)

if __name__ == '__main__':
    
    main(save_idx=37)
    # concatenate figures
    # cat_figures()
    # generate gif file
    # img_paths = [r'visualization\appendix\{:02d}\9.jpg'.format(i) for i in range(40)]
    # img_list = [np.array(Image.open(p)) for p in img_paths]
    # imgs2gif(img_list, 'prediction.gif', duration=0.06 * 10)

