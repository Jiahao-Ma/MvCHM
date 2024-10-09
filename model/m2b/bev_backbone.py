import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseBEVBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config.MODEL
        self.data = config.DATA_CONFIG.DATASET
        input_channels = self.model_cfg.BACKBONE_2D.IN_CHANNELS
        if self.model_cfg.BACKBONE_2D.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.BACKBONE_2D.LAYER_NUMS) == len(self.model_cfg.BACKBONE_2D.LAYER_STRIDES) == len(self.model_cfg.BACKBONE_2D.NUM_FILTERS)
            layer_nums = self.model_cfg.BACKBONE_2D.LAYER_NUMS
            layer_strides = self.model_cfg.BACKBONE_2D.LAYER_STRIDES
            num_filters = self.model_cfg.BACKBONE_2D.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.BACKBONE_2D.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.BACKBONE_2D.UPSAMPLE_STRIDES) == len(self.model_cfg.BACKBONE_2D.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.BACKBONE_2D.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.BACKBONE_2D.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, batch_dict):
        
        compact_spatial_features = batch_dict['spatial_features']
        spatial_features_shape = compact_spatial_features.shape[-2:]
        ups = []
        x = compact_spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                if self.data == 'MultiviewX':
                    ups_feat = F.interpolate(self.deblocks[i](x), size=spatial_features_shape) 
                    ups.append(ups_feat)
                else:
                    ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        batch_dict['compact_spatial_features'] = x # 1, 384, 120, 360

        return batch_dict
