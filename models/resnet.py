from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
from models.backbone.resnet_utils import resnet50
from models.gcn_lib import Grapher, act_layer
from timm.models.layers import DropPath

class Conv_FFN(nn.Module):
    def __init__(self, in_features, mlp_ratio=4, out_features=None, act='GeLU', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = in_features * mlp_ratio
        self.conv1 =nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)

class GNN_Transformer(nn.Module):
    def __init__(
        self,
        dim=256,
        blocks = 1, 
        epsilon = 0.1,
        kernels = [3,5],
        reduction_ration = 1,
        dilation_rate = [1]
    ):        
        super().__init__()
        self.backbone = nn.ModuleList([])
        for j in range(blocks):
            self.backbone += [
                nn.Sequential(Grapher(dim, kernel_size=kernels[j], dilation=dilation_rate[j], conv='edge', act='gelu', norm='batch',
                                bias=True, stochastic=False, epsilon=epsilon , r=reduction_ration, n=196, drop_path=0.0,
                                relative_pos=True),
                      Conv_FFN(dim)
                     )]
       

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        embedding_final = x + residual
        return embedding_final
class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])

class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()  # res5
        self.layer4 = nn.Sequential(resnet.layer4)  # res5
        self.out_channels = [1024, 2048]
        hidden_dim = 256
        in_dim = 1024
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 1, stride=1, padding=0)
        self.conv2= nn.Conv2d(hidden_dim, in_dim, 1, stride=1, padding=0)
        self.graph_transformer = GNN_Transformer(dim=hidden_dim, blocks=1)


    def forward(self, x):
        shortcut = x
        conv1 = self.conv1(x)
        x_gnn_feat=self.graph_transformer(conv1)
        
        conv2 = self.conv2(x_gnn_feat)
       
        layer5_feat = self.layer4(conv2)
 
    
        x_feat = F.adaptive_max_pool2d(conv2, 1)

        feat = F.adaptive_max_pool2d(layer5_feat, 1)
        
        return OrderedDict([["feat_res4", x_feat], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=True):
    # resnet = resnet50(pretrained=True)
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)
    