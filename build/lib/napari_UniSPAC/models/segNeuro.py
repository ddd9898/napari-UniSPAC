import torch
import torch.nn as nn
from models.unet2d import UNet2d
from models.unet3d import UNet3d

from copy import deepcopy



####ACRLSD模型
class ACRLSD(torch.nn.Module):
    def __init__(
        self,
    ):
        super(ACRLSD, self).__init__()

        # create our network, 1 input channels in the raw data
        self.model_lsds = UNet2d(
            in_channels=1, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2],[2,2],[2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.lsd_predict = torch.nn.Conv2d(in_channels=12,out_channels=6, kernel_size=1)  #最终输出层的卷积操作


        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2],[2,2],[2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.affinity_predict = torch.nn.Conv2d(in_channels=12,out_channels=2, kernel_size=1)  #最终输出层的卷积操作
    
    def forward(self, x):

        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x,y_lsds],dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds,y_affinity
    
    
####ACRLSD模型
class segEM2d(torch.nn.Module):
    def __init__(
        self,
    ):
        super(segEM2d, self).__init__()
        
        ##For affinity prediction
        self.model_affinity = ACRLSD()
        model_path = './checkpoints/ACRLSD_2D(hemi+fib25)_Best_in_val.model' 
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        self.model_affinity.load_state_dict(weights)
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 2 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet2d(
            in_channels=3, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2],[2,2],[2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.mask_predict = torch.nn.Conv2d(in_channels=12,out_channels=1, kernel_size=1)  #最终输出层的卷积操作
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x_raw, x_prompt):

        y_lsds,y_affinity = self.model_affinity(x_raw)
        
        y_concat = torch.cat([x_prompt.unsqueeze(1),y_affinity],dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask,y_lsds,y_affinity



####ACRLSD_3d模型

####ACRLSD模型
class ACRLSD_3d(torch.nn.Module):
    def __init__(
        self,
    ):
        super(ACRLSD_3d, self).__init__()
        # d_factors = [[2,2,2],[2,2,2],[2,2,2]]  #降采样的因子
        # in_channels=1 #输入的图像通道数
        # num_fmaps=12
        # fmap_inc_factor=5

        # create our network, 1 input channels in the raw data
        self.model_lsds = UNet3d(
            in_channels=1, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.lsd_predict = torch.nn.Conv3d(in_channels=12,out_channels=10, kernel_size=1)  #最终输出层的卷积操作


        # create our network, 10 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet3d(
            in_channels=11, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.affinity_predict = torch.nn.Conv3d(in_channels=12,out_channels=3, kernel_size=1)  #最终输出层的卷积操作
    
    def forward(self, x):
        
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x,y_lsds.detach()],dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds,y_affinity

    

class segEM3d(torch.nn.Module):
    def __init__(
        self,
    ):
        super(segEM3d, self).__init__()
        
        ##2D slice mask prediction
        self.model_mask_2d = segEM2d()
        model_path = './checkpoints/segEM2d(hemi+fib25)_Best_in_val.model'
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        self.model_mask_2d.load_state_dict(weights)
        for param in self.model_mask_2d.parameters():
            param.requires_grad = False
        
        
        ##For affinity prediction
        self.model_affinity = ACRLSD_3d()
        model_path = './checkpoints/ACRLSD_3D(hemi+fib25)_Best_in_val.model' 
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        self.model_affinity.load_state_dict(weights)
        for param in self.model_affinity.parameters():
            param.requires_grad = False
            

        # create our network, 3 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet3d(
            in_channels=4, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.mask_predict = torch.nn.Conv3d(in_channels=12,out_channels=1, kernel_size=1)  #最终输出层的卷积操作
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x_raw, x_prompt):
        '''
        x_raw: shape = (Batch * channel * dim_x * dim_y * dim_z)
        x_prompt: shape = (Batch * channel * dim_x * dim_y * dim_z)
        '''
        ##Get mask for slice0
        y_mask2d_slice0,_,_ = self.model_mask_2d(x_raw[:,:,:,:,0],x_prompt)
        
        ##Get affinity for raw
        y_lsds,y_affinity = self.model_affinity(x_raw)
        
        #replace raw slice0
        x_raw_new = deepcopy(x_raw)
        x_raw_new[:,0,:,:,0] = (y_mask2d_slice0.detach().squeeze()>0.5) + 0
        y_concat = torch.cat([x_raw_new,y_affinity.detach()],dim=1)

        y_mask3d = self.mask_predict(self.model_mask(y_concat))
        y_mask3d = self.sigmoid(y_mask3d)

        return y_mask3d,y_mask2d_slice0,y_affinity,y_lsds

    
    
class segEM_3d_trace(torch.nn.Module):
    def __init__(
        self,
    ):
        super(segEM_3d_trace, self).__init__()
        
        ##For affinity prediction
        self.model_affinity = ACRLSD_3d()
        model_path = './checkpoints/ACRLSD_3D(hemi+fib25)_Best_in_val.model' 
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        self.model_affinity.load_state_dict(weights)
        for param in self.model_affinity.parameters():
            param.requires_grad = False
            

        # create our network, 3 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet3d(
            in_channels=4, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.mask_predict = torch.nn.Conv3d(in_channels=12,out_channels=1, kernel_size=1)  #最终输出层的卷积操作
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x_raw, gt_mask2d_slice0):
        '''
        x_raw: shape = (Batch * channel * dim_x * dim_y * dim_z)
        gt_mask2d_slice0: shape = (Batch * dim_x * dim_y)
        '''
        # ##Get mask for slice0
        # y_mask2d_slice0,_,_ = self.model_mask_2d(x_raw[:,:,:,:,0],x_prompt)
        
        ##Get affinity for raw
        y_lsds,y_affinity = self.model_affinity(x_raw)
        
        #replace raw slice0
        x_raw_new = deepcopy(x_raw)
        x_raw_new[:,0,:,:,0] = gt_mask2d_slice0
        y_concat = torch.cat([x_raw_new,y_affinity.detach()],dim=1)

        y_mask3d = self.mask_predict(self.model_mask(y_concat))
        y_mask3d = self.sigmoid(y_mask3d)

        return y_mask3d,y_affinity,y_lsds
