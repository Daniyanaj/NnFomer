"""
Discriminator Network Definition

"""
import torchvision.transforms as T
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch
from torchvision import transforms
from einops import rearrange

NORM_EPS = 1e-5
from functools import partial


class LayerNorm_X4(nn.Module):
    def __init__(self,  dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 4 * 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        D, H, W = self.input_resolution
        x = x.permute(0, 4, 1, 2, 3)
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale,
                      p3=self.dim_scale,
                      c=C // (self.dim_scale ** 3))
        # x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, module_type, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, dropout_p=0.0, norm=True, activation=True):
        super().__init__()
        if module_type == 'convolution':
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        else:
            raise NotImplementedError(f"Module type '{module_type}' is not valid")

        self.lrelu = nn.LeakyReLU(0.2) if activation else None
        self.norm = nn.LayerNorm(out_channels) if norm else None

    def forward(self, x):
        x = self.lrelu(x) if self.lrelu else x
        x = self.conv(x)
        
        if self.norm:
            B, C, D, H, W = x.shape
            x =x.permute(0,2,3,4,1)
            x = self.norm(x)
            x = x.permute(0,4,1,2,3)
        else:
            x = x
        return x


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, in_channels, out_channels, head_dim=8):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1,
                                       padding=1, groups=in_channels // head_dim, bias=False)
        self.norm = norm_layer(in_channels)
        self.act = nn.ReLU(inplace=True) 
        self.projection = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        out = out + x
        out = self.proj(out)
        return out

class MHCA_1(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, in_channels, out_channels, head_dim=8):
        super(MHCA_1, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=True)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True) 
        self.projection = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=True)
        

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        out = out   
        return out
    
class PatchGAN(nn.Module):
    def __init__(self, in_channels=1593, out_channels=1, bias=False, norm=True):
        super().__init__()
        self.dim_inter1 = in_channels-5
        self.conv1 = nn.Conv3d(1593, 1536, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(1536, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.MHCA_1_conv1 = MHCA_1(self.dim_inter1, self.dim_inter1)
        # self.MHCA_1_conv2 = MHCA_1(256, 256)
        self.dim_inter = in_channels + 9
        self.dim_inter1 = in_channels -1 + 4
        self.conv0 = nn.Conv3d(1593, 1593,   kernel_size=3, stride=1, padding=1, bias=False)
        
        #self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1)
        
# =============================================================================
#         self.discriminator_blocks = nn.ModuleList([
#             DiscriminatorBlock('convolution', 256, 64, bias=bias, norm=False, activation=False),
#             DiscriminatorBlock('convolution', 64, 128, bias=bias, norm=True, activation=True),
#             DiscriminatorBlock('convolution', 128, 256, bias=bias, norm=True, activation=True),
#             DiscriminatorBlock('convolution', 256, 512, bias=bias, norm=True, activation=True),
#             DiscriminatorBlock('convolution', 512, out_channels, bias=bias, norm=False, stride=1)
#         ])
# =============================================================================
        self.discriminator_blocks = nn.ModuleList([
            MHCA(256, 64),
            MHCA(64, 128),
            MHCA(128, 256),
            MHCA(256, 512),
            DiscriminatorBlock('convolution', 512, out_channels, bias=bias, norm=False, stride=1)
        ])
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, mask, backbone, modalities=False):
        try:
            
            B,C,D,H,W = backbone.shape
            down = torch.nn.Upsample(size=(D, H, W))
            img_down = down(img)
            mask_down = down(mask[0])
            mask_down_1=down(mask[1])
            mask_down_2=down(mask[2])
            
            output = torch.cat([img_down, backbone, mask_down,mask_down_1,mask_down_2], 1)
        except:
            print("size mismatch!!")
            print("OMKAR $$$$$$$$")

            return None
        if modalities:
            output = self.conv0(output)
        output = self.conv1(output)

        output = output + backbone
        
        output = self.conv2(output)
        
        for block in self.discriminator_blocks:
            output = block(output)
        
        output = self.sigmoid(output)
        return output
class PatchGAN_tar(nn.Module):
    def __init__(self, in_channels=1593, out_channels=1, bias=False, norm=True):
        super().__init__()
        self.dim_inter1 = in_channels-5
        self.conv1 = nn.Conv3d(1540, 1536, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(1536, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.MHCA_1_conv1 = MHCA_1(self.dim_inter1, self.dim_inter1)
        # self.MHCA_1_conv2 = MHCA_1(256, 256)
        self.dim_inter = in_channels + 9
        self.dim_inter1 = in_channels -1 + 4
        self.conv0 = nn.Conv3d(1540, 1540,   kernel_size=3, stride=1, padding=1, bias=False)
        
        #self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1)
        
# =============================================================================
#         self.discriminator_blocks = nn.ModuleList([
#             DiscriminatorBlock('convolution', 256, 64, bias=bias, norm=False, activation=False),
#             DiscriminatorBlock('convolution', 64, 128, bias=bias, norm=True, activation=True),
#             DiscriminatorBlock('convolution', 128, 256, bias=bias, norm=True, activation=True),
#             DiscriminatorBlock('convolution', 256, 512, bias=bias, norm=True, activation=True),
#             DiscriminatorBlock('convolution', 512, out_channels, bias=bias, norm=False, stride=1)
#         ])
# =============================================================================
        self.discriminator_blocks = nn.ModuleList([
            MHCA(256, 64),
            MHCA(64, 128),
            MHCA(128, 256),
            MHCA(256, 512),
            DiscriminatorBlock('convolution', 512, out_channels, bias=bias, norm=False, stride=1)
        ])
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, mask, backbone, modalities=False):
        try:
            
            B,C,D,H,W = backbone.shape
            down = torch.nn.Upsample(size=(D, H, W))
            img_down = down(img)
            mask_down = down(mask[0])
            mask_down_1=down(mask[1])
            mask_down_2=down(mask[2])
            
            output = torch.cat([img_down, backbone, mask_down,mask_down_1,mask_down_2], 1)
        except:
            print("size mismatch!!")
            print("OMKAR $$$$$$$$")

            return None
        if modalities:
            output = self.conv0(output)
        output = self.conv1(output)
        output = output + backbone
        
        output = self.conv2(output)
        
        for block in self.discriminator_blocks:
            output = block(output)
        
        output = self.sigmoid(output)
        return output