from distutils.file_util import copy_file
import os
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from zmq import device; matplotlib.use('agg')
from lib.utils.tool_utils import gen_scale
from tensorboardX import SummaryWriter
from distutils.dir_util import copy_tree
from shutil import copyfile
from lib.data.multiviewX import MultiviewX
from lib.data.wildtrack import Wildtrack
from lib.data.dataloader import MultiviewDataset, collater, get_padded_value                           
from lib.utils.config_utils import cfg, cfg_from_yaml_file
from lib.utils.tool_utils import MetricDict, to_numpy
from lib.utils.visual_utils import visualize_heatmap, reverse_image, visualize_kyp_heatmap
from model.stem.mvchm import MvCHM
from lib.utils.visual_utils import Process, Monitor
from model.loss.loss import compute_loss

class Trainer(object):
    def __init__(self, model, args, device, summary, loss_weight=[1., 1.]) -> None:
        self.model = model
        self.args = args
        self.device = device
        self.summary = summary
        self.loss_weight = loss_weight
        self.monitor = Monitor()
    def train(self, dataloader, optimizer, epoch, args):
        self.model.train()
        epoch_loss = MetricDict()
        t_b = time.time()
        t_forward, t_backward = 0, 0
        with tqdm(total=len(dataloader), desc=f'\033[33m[TRAIN]\033[0m Epoch {epoch} / {args.epochs}', postfix=dict, mininterval=0.2) as pbar:
            for idx, data in enumerate(dataloader):
                
                batch_dict, batch_pred = self.model(data)
                
                t_f = time.time()
                t_forward += t_f - t_b
                
                loss, loss_dict = compute_loss(batch_pred, data, self.loss_weight)
                
                epoch_loss += loss_dict
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t_b = time.time()
                t_backward += t_b - t_f

                if idx % args.print_iter == 0:
                    mean_loss = epoch_loss.mean
                    pbar.set_postfix(**{
                        '(1)loss_total' : '\033[33m{:.6f}\033[0m'.format(mean_loss['loss']),
                        '(2)loss_heatmap' : '{:.5}'.format(mean_loss['loss_heatmap']),
                        '(4)t_f & t_b' : '{:.2f} & {:.2f}'.format(t_forward/(idx+1), t_backward/(idx+1))
                        }
                    )
                    pbar.update(1)
                if idx % args.vis_iter == 0:
                    steps = (epoch-1) * (len(dataloader) // args.vis_iter) + idx // args.vis_iter
                    heatmap_fig = visualize_heatmap(pred=torch.sigmoid(batch_pred['heatmap']).squeeze(dim=0).squeeze(dim=-1), gt=data['heatmap'])
                    self.summary.add_figure('train/heatmap', heatmap_fig, steps)
                    monitor_fig = self.monitor.visualize(batch_dict, batch_pred, data, show=False)
                    monitor_fig[0].savefig("projection_res.jpg")
                    self.summary.add_figure('train/monitor', monitor_fig[0], steps)
            
        return epoch_loss.mean
    

    def validate(self, dataloader, epoch, args):
        self.model.eval()
        epoch_loss = MetricDict()
        t_b = time.time()
        t_forward, t_backward = 0, 0
        with tqdm(total=len(dataloader), desc=f'\033[31m[VAL]\033[0m Epoch {epoch} / {args.epochs}', postfix=dict, mininterval=0.2) as pbar:
            for idx, data in enumerate(dataloader):
                with torch.no_grad():

                    batch_dict, batch_pred = self.model(data)
                    
                    t_f = time.time()
                    t_forward += t_f - t_b
                    
                    loss, loss_dict = compute_loss(batch_pred, data, self.loss_weight)
                    
                    epoch_loss += loss_dict

                    t_b = time.time()
                    t_backward += t_b - t_f

                    if idx % args.print_iter == 0:
                        mean_loss = epoch_loss.mean
                        pbar.set_postfix(**{
                            '(1)loss_total' : '\033[33m{:.6f}\033[0m'.format(mean_loss['loss']),
                            '(2)loss_heatmap' : '{:.5}'.format(mean_loss['loss_heatmap']),
                            '(4)t_f & t_b' : '{:.2f} & {:.2f}'.format(t_forward/(idx+1), t_backward/(idx+1))
                            }
                        )
                        pbar.update(1)
        return epoch_loss.mean

def make_lr_scheduler(optimizer):
    w_iters = 5
    w_fac = 0.1
    max_iter = 40
    lr_lambda = lambda iteration : w_fac + (1 - w_fac) * iteration / w_iters \
            if iteration < w_iters \
            else 1 - (iteration - w_iters) / (max_iter - w_iters)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
    return scheduler

def setup_seed(seed=7777):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def make_experiment(args, cfg, copy_repo=False):
    lastdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.DATA_CONFIG.DATASET == 'Wildtrack':
        lastdir += '_wt'
    elif cfg.DATA_CONFIG.DATASET == 'MultiviewX':
        lastdir += '_mx'
    args.savedir = os.path.join(args.savedir , lastdir)
    summary = SummaryWriter(args.savedir+'/tensorboard')
    summary.add_text('config', '\n'.join(
        '{:12s} {}'.format(k, v) for k, v in sorted(args.__dict__.items())))
    summary.file_writer.flush()
    if copy_repo:
        os.makedirs(args.savedir, exist_ok=True)
        copy_file(args.cfg_file, args.savedir)
    return summary, args

def resume_experiment(args):
    summary_dir = os.path.join(args.savedir, args.resume, 'tensorboard')
    args.savedir = os.path.join(args.savedir, args.resume)
    summary = SummaryWriter(summary_dir)
    return summary, args

def save(model, epoch, args, optimizer, scheduler, train_loss, val_loss):
    savedir = os.path.join(args.savedir, 'checkpoints')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    checkpoints = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'args':args
    }
    torch.save(checkpoints, os.path.join(savedir, 'Epoch{:02d}_train_loss{:.4f}_val_loss{:.4f}.pth'.\
                        format(epoch, train_loss['loss'], val_loss['loss'])))

def resume(resume_dir, model, optimizer, scheduler, load_model_ckpt_only=False):
    checkpoints = torch.load(resume_dir)
    pretrain = checkpoints['model_state_dict']
    current = model.state_dict()
    state_dict = {k: v for k, v in pretrain.items() if k in current.keys()}
    current.update(state_dict)
    model.load_state_dict(current)
    if load_model_ckpt_only:
        return model, None, None, 1
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    epoch = checkpoints['epoch'] + 1
    print("Model resume training from %s" %resume_dir)
    return model, optimizer, scheduler, epoch

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=r'F:\ANU\ENGN8602\Code\MvDDE\MvCHM\cfgs\MvDDE.yaml',\
         help='specify the config for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')         
    
     # Training options
    parser.add_argument('-e', '--epochs', type=int, default=40,
                        help='the number of epochs for training')

    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for training. [NOTICE]: this repo only support \
                              batch size of 1')

    parser.add_argument('--lr', type=float, default=0.0002,#0.0002,
                        help='learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='learning rate')   

    parser.add_argument('--lr_step', type=list, default=[90, 120],
                        help='learning step')

    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='learning factor')
    
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')                        

    parser.add_argument('--savedir', type=str,
                        default='experiments')   

    parser.add_argument('--resume', type=str,
                        default=None)
    
    parser.add_argument('--checkpoint', type=str,
                        default=None)

    parser.add_argument('--print_iter', type=int, default=1,
                        help='print loss summary every N iterations')

    parser.add_argument('--vis_iter', type=int, default=30,
                        help='display visualizations every N iterations')      

    parser.add_argument('--loss_weight', type=float, default=[1., 1.],
                        help= '2D weight of each loss only including heatmap and location.')        

    parser.add_argument('--copy_yaml', type=bool, default=True,
                        help='Copy the whole repo before training')
    


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    # define devices
    device = torch.device('cuda:0')

    # define preprocess operation and dataloader
    new_w, new_h, old_w, old_h = cfg.DATA_CONFIG.NEW_WIDTH, cfg.DATA_CONFIG.NEW_HEIGHT, cfg.DATA_CONFIG.OLD_WIDTH, cfg.DATA_CONFIG.OLD_HEIGHT
    scale, scale_h, scale_w = gen_scale(new_w, new_h, old_w, old_h)
    pad_h, pad_w = get_padded_value(new_h, new_w)    

    process = Process(scale_h, scale_w, pad_h, pad_w, new_h, new_w, old_h, old_w)

    # mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=torch.float32)
    # std = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=torch.float32)

    dataname = 'Wildtrack'
    path = 'F:\ANU\ENGN8602\Data\{}'.format(dataname) # MultiviewX
    DATASET = {'MultiviewX': MultiviewX, 'Wildtrack': Wildtrack}
    if dataname == 'MultiviewX':
        detector_ckpt = r'model\detector\checkpoint\rcnn_mxp.pth'
        keypoint_ckpt = r'model\refine\checkpoint\mspn_mx.pth'
    elif dataname == 'Wildtrack':
        detector_ckpt = r'model\detector\checkpoint\rcnn_wtp.pth'
        keypoint_ckpt = r'model\refine\checkpoint\mspn_wt.pth'
        
    dataset_val = MultiviewDataset( DATASET[dataname](root=path), set_name='val')
    val_dataloader = DataLoader( dataset_val, num_workers=1, batch_size=1, collate_fn=collater)

    dataset_train = MultiviewDataset( DATASET[dataname](root=path), set_name='train')
    train_dataloader = DataLoader( dataset_train, num_workers=1, batch_size=1, collate_fn=collater)

    # define model
    model = MvCHM(cfg, dataset_train, process, device,
                  detector_ckpt=detector_ckpt,
                  keypoint_ckpt=keypoint_ckpt)

    optimizer = optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, args.lr_step, args.lr_factor )

    # Create Summary & Resume Training
    if args.resume is not None:
        summary, args = resume_experiment(args)
        resume_dir = os.path.join(args.savedir, 'checkpoints', args.checkpoint)
        # resume_dir = args.checkpoint
        model, optimizer, scheduler, start = \
            resume(resume_dir, model, optimizer, scheduler)
        args.epochs = args.epochs + 5
    else:
        summary, args = make_experiment(args, cfg, args.copy_yaml)
        start = 1
        
    trainer = Trainer(model, args, device, summary, args.loss_weight)

    for epoch in range(start, args.epochs+1):
        summary.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Train model
        train_loss = trainer.train(train_dataloader, optimizer, epoch, args) 

        # Evaluate model
        val_loss = trainer.validate(val_dataloader, epoch, args)

        summary.add_scalars('loss', {'train_loss': train_loss['loss'], 'val_loss' : val_loss['loss']}, epoch)

        scheduler.step()

        if epoch % 1 == 0:
            save(model, epoch, args, optimizer, scheduler, train_loss, val_loss)

if __name__ == '__main__':
    main()
