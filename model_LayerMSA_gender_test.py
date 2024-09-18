from this import d
from tkinter import X
from unittest.mock import patch
import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
# from dataset import BATCH_SIZE

# from visualizer import get_local

EMBEDDING_DIM = 256

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob

    # Here, shape is of type tuple, the first element is x.shape[0], followed by (dimension of x - 1) 1s ->: (n,1,1,1,...)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)


    random_tensor.floor_()  # binarize

    # ???
    output = x.div(keep_prob) * random_tensor
    return output

# A class of random depth, calling the random depth function above
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    # The embedding dimension of the VIT_base model is 768. If it is a larger model, the embedding depth will be larger.
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=EMBEDDING_DIM, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        # Calculate how many patches to divide an image into
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Embedding using 2D convolution, input 3 channels, output 768 channels; step size = patch_size;
        # Dimension from (224*224*3) -> (14*14*768)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        # If norm_layer is not passed in, no operation will be performed
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape


        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]

        # flatten从第二个维度开始,也就是H开始;再交换1,2两个维度
        x = self.proj(x).flatten(2).transpose(1, 2) 
        
        # x += self.pos_encoding

        x = self.norm(x) 
        return x

# Module for implementing Multi-head self-attention
class Attention(nn.Module):
    def __init__(self,
                 dim,   # token dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        # In the case of multiple heads, the dimension of each small head = input dimension / number of heads
        head_dim = dim // num_heads

        # After Q is multiplied by the transpose of K, it needs to be divided by a square root dk. The formula in the theory is introduced
        self.scale = qk_scale or head_dim ** -0.5 #

        # Generate three Q, K, V matrices through a fully connected layer (some codes use three fully connected layers to generate them separately)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop_ratio)

        # This fully connected layer is used to concatenate the results of multiple heads
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    # @get_local('attn_map')
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # 这是传入x的维度;第0维是bs;第1维是有多少个patches,这里224/16=14,所以14*14=196个patches,+1是一个class_token;第2维数值上就是768;
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim],先经过qkv这个全连接层,将最后的维度数变成三倍;

        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head],再做这个reshape操作;
        # 3:因为有qkv三个矩阵;self.num_heads:有多少个头; C//self.num_heads:每一个头对应的qkv的维度;

        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]，这一步之后q，k，v的维度就变成这样了;把他们单独取出来了(切片)；
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]：将k矩阵最后两个维度进行转置，再和q做矩阵乘法；
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]：此时乘法之后的最后两个维度就变成这样；之后再除以一个根号dk的值；
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 这里dim=-1就是针对每一行做一个softmax处理
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_map = attn

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]：attention乘以矩阵v之后就是这个维度；
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]：将最后两个维度的信息拼接到一起，就是很多个head的结果concate到一起的操作；
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  #再经过投影
        x = self.proj_drop(x)
        return x

# ViT_B16模型中的mlp部分，很简单
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    # hidden_features一般是in_features的四倍(看图)
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ViT_B16中的Encoder_Block部分(看图)
class Block(nn.Module):
    def __init__(self,
                 dim,       #每个token的dimention
                 num_heads,     #Multi_head self_attention中使用的header个数
                 mlp_ratio=4.,  #第一个全连接层的输出个数是输入的四倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0., #Multi_head self_attention中最后的全连接层使用的drop_ratio
                 attn_drop_ratio=0.,    #q乘k.T之后通过的softmax之后有一个dropout层的drop_ratio(图上好像看不到,代码里面有)
                 drop_path_ratio=0.,    #模型图中Encoder_Block中对应的那两层drop_out层对应的drop_ratio
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                #  norm_layer = nn.Identity
                 ):
        super(Block, self).__init__()

        # 结构图中第一个layer_norm
        self.norm1 = norm_layer(dim)

        # 调用刚刚定义的attention创建一个Multi_head self_attention模块
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        # 使不用使用DropPath方法
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        # 结构图中第二个layer_norm
        self.norm2 = norm_layer(dim)

        # MLP中第一个全连接层的输出神经元的倍数(4倍)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # 前向传播，很清晰
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))


        return x


class self_attention(nn.Module):
    def __init__(self,
                dim,
                num_heads):
        super(self_attention,self).__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim,dim*3,bias=True)

        self.proj = nn.Linear(dim,dim)

        self.avgpooling = nn.AdaptiveAvgPool2d((16,16))

    def forward(self,x):

        x = self.avgpooling(x)

        B, C, W, H = x.shape

        S = W * H

        x = x.reshape(B,C,S)

        qkv = self.qkv(x).reshape(B, C, 3, self.num_heads, S // self.num_heads).permute(2,0,3,1,4)

        q, k, v =qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1)) * self.scale

        attn = attn.softmax(dim=-1)
        
        # x = (attn @ v).transpose(1,2).reshape(B,C,W,H)
        x = (attn @ v).transpose(1,2).reshape(B,C,-1)

        return x




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Res_Vit_B16(nn.Module):
    """
        Resnet最后的fc层参数改了;
        layer4和AveragePooling中间加了:PatchEmbed、Encoder Blocks * depth;

    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer_res=None, in_channels=3,\

                 img_size=10,patch_size=2,in_c=512,embed_dim=768,depth=12,num_heads=12,
                 mlp_ratio=4.0,qkv_bias=True,qk_scale=None,representation_size=None,
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,norm_layer_vit=None,act_layer=None
                ):
        super(Res_Vit_B16, self).__init__()
        
        if norm_layer_res is None:
            norm_layer_res = nn.BatchNorm2d
        self._norm_layer_res = norm_layer_res

        self.num_classed = num_classes
        norm_layer_vit = norm_layer_vit or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer_res(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第一个layer不会下采样，因为stride默认为1；后面三个layer的stride都是2；
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.patch_embed1 = embed_layer(img_size=80,patch_size=16,in_c=64,embed_dim=embed_dim)

        self.block1 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)
        


        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.patch_embed2 = embed_layer(img_size=40,patch_size=8,in_c=128,embed_dim=embed_dim)

        self.block2 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)



        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.patch_embed3 = embed_layer(img_size=20,patch_size=4,in_c=256,embed_dim=embed_dim)

        self.block3 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)



        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.patch_embed4 = embed_layer(img_size=10,patch_size=2,in_c=in_c,embed_dim=embed_dim)

        self.block4 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)
        
        # self.attn = Attention(dim=embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
        #                       attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=drop_ratio)

        self.patch_embed = embed_layer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            # 在一个列表里面用for循环创建depth(12)个Encoder_Block,之后再用nn.Sequential()方法全部打包起来
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], # ratio递增
                  norm_layer=norm_layer_vit, act_layer=act_layer)
            for i in range(depth)
        ])


        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((self.patch_embed.num_patches,1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(self.patch_embed.num_patches * block.expansion, num_classes)

        self.add = nn.Linear(num_classes,1)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    # 这里传入的参数block是Basci或者Bottleneck结构，blocks是创建Resnet的时候传入的[3，4，6，4]其中的一个具体的值，上面用layer[i]传入的；
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer_res
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 这里stride=2，因为上面传入的是2；所以这一层减少了fm的维度
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        # 这里stride=1，使用默认值；所以这里相当于没有做下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        # 前面4层是静态的，所有Resnet的结构都要经过这四层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 后四层是动态的，也是Res18、Res34、Res50、Res101的区别
        x = self.layer1(x)  # 经过leyer1 ：w * H = 80 * 80
        out1 = self.patch_embed1(x)
        out1 = self.block1(out1)

        x = self.layer2(x)  # 经过leyer2 ：w * H = 40 * 40
        out2 = self.patch_embed2(x)
        out2 = self.block2(out2)

        x = self.layer3(x)  # 经过leyer3 ：w * H = 20 * 20
        out3 = self.patch_embed3(x)
        out3 = self.block3(out3)

        x = self.layer4(x)  # 经过leyer4 ：W * H = 10 * 10
        out4 = self.patch_embed4(x)
        out4 = self.block4(out4)

        # x = self.patch_embed(x)

        # x = self.blocks(x)

        x = out1 + out2 + out3 + out4
        
        x = self.avgpool(x)
        # print(x.shape)

        # 这是按照列来拼接
        x = torch.flatten(x, 1)

        # x = x.permute(1,0)
        x = self.fc(x)

        x = self.add(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)




class Res_Vit_B16_gender_512_new(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer_res=None, in_channels=3,\
                 
                 img_size=10,patch_size=[16, 8, 4, 2],
                 in_c=[64,128,256,512],embed_dim=EMBEDDING_DIM,depth=12, # EMBEDDING_DIM是个全局变量,要么取256,要么取768
                #  num_heads=12, 这里注意一下，改过的
                 num_heads=8,
                 mlp_ratio=4.0,qkv_bias=True,qk_scale=None,representation_size=None,
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,norm_layer_vit=None,act_layer=None,
                 num_sa = 4
                ):
        super(Res_Vit_B16_gender_512_new, self).__init__()
        
        if norm_layer_res is None:
            norm_layer_res = nn.BatchNorm2d
        self._norm_layer_res = norm_layer_res

        self.num_classed = num_classes
        norm_layer_vit = norm_layer_vit or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.inplanes = 64
        self.dilation = 1
        self.num_sa = num_sa
        # self.block = block
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer_res(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第一个layer不会下采样，因为stride默认为1；后面三个layer的stride都是2；
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.patch_embed1 = embed_layer(img_size=128,patch_size=patch_size[0],in_c=in_c[0],embed_dim=embed_dim)

        self.block1 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=drop_path_ratio,
                            norm_layer=norm_layer_vit,act_layer=act_layer)
        


        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.patch_embed2 = embed_layer(img_size=64,patch_size=patch_size[1],in_c=in_c[1],embed_dim=embed_dim)

        self.block2 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=drop_path_ratio,
                            norm_layer=norm_layer_vit,act_layer=act_layer)



        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.patch_embed3 = embed_layer(img_size=32,patch_size=patch_size[2],in_c=in_c[2],embed_dim=embed_dim)

        self.block3 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=drop_path_ratio,
                            norm_layer=norm_layer_vit,act_layer=act_layer)



        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.patch_embed4 = embed_layer(img_size=16,patch_size=patch_size[3],in_c=in_c[3],embed_dim=embed_dim)

        self.block4 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=drop_path_ratio,
                            norm_layer=norm_layer_vit,act_layer=act_layer)
        
        # self.attn = Attention(dim=embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
        #                       attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=drop_ratio)

        # self.patch_embed = embed_layer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            # 在一个列表里面用for循环创建depth(12)个Encoder_Block,之后再用nn.Sequential()方法全部打包起来
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], # ratio递增
                  norm_layer=norm_layer_vit, act_layer=act_layer)
            for i in range(depth)
        ])

        self.fc_y = nn.Linear(in_c[3],768)
        # self.fc_y_add1 = nn.Linear(256,128)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((self.patch_embed1.num_patches * self.num_sa, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(self.patch_embed1.num_patches * self.num_sa + 1, num_classes)

        self.add1 = nn.Linear(num_classes,1024)
        self.relu = nn.ReLU(inplace=False) 

        self.add2 = nn.Linear(1024,1)

        # self.add3 = nn.Linear(10,1)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    # 这里传入的参数block是Basci或者Bottleneck结构，blocks是创建Resnet的时候传入的[3，4，6，4]其中的一个具体的值，上面用layer[i]传入的；
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer_res
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        # 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 这里stride=2，因为上面传入的是2；所以这一层减少了fm的维度
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        # 这里stride=1，使用默认值；所以这里相当于没有做下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_gender):
        # See note [TorchScript super()]

        # 前面4层是静态的，所有Resnet的结构都要经过这四层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 后四层是动态的，也是Res18、Res34、Res50、Res101的区别
        x = self.layer1(x)  
        y = x
        out1 = self.patch_embed1(x)
        out1 = self.block1(out1)

        x = self.layer2(y)  
        y = x
        out2 = self.patch_embed2(x)
        out2 = self.block2(out2)

        x = self.layer3(y) 
        y = x
        out3 = self.patch_embed3(x)
        out3 = self.block3(out3)

        x = self.layer4(y)  
        y = x
        out4 = self.patch_embed4(x)
        out4 = self.block4(out4)

        # x = self.patch_embed(x)

        # x = self.blocks(x)

        bs, c, _, _ = y.shape
        y = y.reshape(bs,c,-1).permute(0,2,1)
        y = self.fc_y(y)

        x = torch.concat((out1,out2,out3,out4),dim=1)
        # x = torch.concat((out3,out3,out4,out4),dim=1)
        # x = torch.concat((out4,out4,out4,out4),dim=1)


        # 为了使用其中部分SA改变y的第二维度,下面才能concate
        # y = y.permute(0,2,1)
        # y = self.fc_y_add1(y)
        # y = y.permute(0,2,1)

        # 用一个残差加起来
        x = x + y
        # x = y
        
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)

        # x = x.permute(1,0)

        x_gender = torch.unsqueeze(x_gender,1)
        x = torch.concat((x,x_gender),dim=-1)

        x = self.fc(x)

        x = self.add1(x)

        x = self.relu(x)

        x = self.add2(x)

        # x = self.relu(x)

        # x = self.add3(x)
        
        return x

    def forward(self, x, x_gender):
        return self._forward_impl(x, x_gender)


# 去掉patchembedding的模型
class Res_Vit_B16_gender_512_sa1(nn.Module):
    """
        Resnet最后的fc层参数改了;
        layer4和AveragePooling中间加了:PatchEmbed、Encoder Blocks * depth;

    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer_res=None, in_channels=3,\
                 num_heads=8,
                 num_sa = 4
                ):
        super(Res_Vit_B16_gender_512_sa1, self).__init__()

        self.num_classed = num_classes
        self.inplanes = 64
        self.dilation = 1

        if norm_layer_res is None:
            norm_layer_res = nn.BatchNorm2d
        self._norm_layer_res = norm_layer_res
        
        self.num_sa = num_sa
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer_res(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第一个layer不会下采样，因为stride默认为1；后面三个layer的stride都是2；
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.sa = self_attention(dim=EMBEDDING_DIM,num_heads=num_heads)


        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])


        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        self.fc_y = nn.Linear(512,960)
        self.avgpool = nn.AdaptiveAvgPool2d((960,1))
        self.fc = nn.Linear(961,num_classes)
        # self.fc_y_add1 = nn.Linear(256,128)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((self.patch_embed1.num_patches * self.num_sa, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(self.patch_embed1.num_patches * self.num_sa * block.expansion + 1, num_classes)

        self.add1 = nn.Linear(num_classes,256)
        self.relu = nn.ReLU(inplace=False) 

        self.add2 = nn.Linear(256,1)

        # self.add3 = nn.Linear(10,1)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    # 这里传入的参数block是Basci或者Bottleneck结构，blocks是创建Resnet的时候传入的[3，4，6，4]其中的一个具体的值，上面用layer[i]传入的；
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer_res
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 这里stride=2，因为上面传入的是2；所以这一层减少了fm的维度
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        # 这里stride=1，使用默认值；所以这里相当于没有做下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_gender):
        # See note [TorchScript super()]

        # 前面4层是静态的，所有Resnet的结构都要经过这四层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 后四层是动态的，也是Res18、Res34、Res50、Res101的区别
        x = self.layer1(x)  # 经过leyer1 ：w * H = 80 * 80
        y = self.layer1(x)
        out1 = self.sa(y)

        x = self.layer2(x)  # 经过leyer2 ：w * H = 40 * 40
        y = self.layer2(y)
        out2 = self.sa(y)
        

        x = self.layer3(x)  # 经过leyer3 ：w * H = 20 * 20
        y = self.layer3(y)
        out3 = self.sa(y)
        

        x = self.layer4(x)  # 经过leyer4 ：W * H = 10 * 10
        y = self.layer4(y)
        out4 = self.sa(y)
        

        # x = self.patch_embed(x)

        # x = self.blocks(x)

        B, C, _, _ = y.shape
        y = y.reshape(B,C,-1).permute(0,2,1)
        y = self.fc_y(y)
        y = y.permute(0,2,1)

        x = torch.concat((out1,out2,out3,out4),dim=1)
        # x = torch.concat((out3,out3,out4,out4),dim=1)
        # x = torch.concat((out4,out4,out4,out4),dim=1)


        # 用一个残差加起来
        x = x + y
        # x = y
        
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)

        # x = x.permute(1,0)

        x_gender = torch.unsqueeze(x_gender,1)
        x = torch.concat((x,x_gender),dim=-1)

        x = self.fc(x)

        x = self.add1(x)

        x = self.relu(x)

        x = self.add2(x)

        # x = self.relu(x)

        # x = self.add3(x)
        
        return x

    def forward(self, x, x_gender):
        return self._forward_impl(x, x_gender)



# 去掉patchembedding并使用bottleneck的模型
class Res_Vit_B16_gender_512_sa1_bottleneck(nn.Module):
    """
        Resnet最后的fc层参数改了;
        
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer_res=None, in_channels=3,\
                 embedding_dim = EMBEDDING_DIM,
                 num_heads=8,
                 num_sa = 4
                ):
        super(Res_Vit_B16_gender_512_sa1_bottleneck, self).__init__()

        self.num_classed = num_classes
        self.inplanes = 64
        self.dilation = 1

        if norm_layer_res is None:
            norm_layer_res = nn.BatchNorm2d
        self._norm_layer_res = norm_layer_res
        
        self.num_sa = num_sa
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer_res(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第一个layer不会下采样，因为stride默认为1；后面三个layer的stride都是2；
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.sa = self_attention(dim=embedding_dim,num_heads=num_heads)


        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])


        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        self.fc_y = nn.Linear(2048,3840)
        self.avgpool = nn.AdaptiveAvgPool2d((3840,1))
        self.fc = nn.Linear(3841,num_classes)
        # self.fc_y_add1 = nn.Linear(256,128)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((self.patch_embed1.num_patches * self.num_sa, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(self.patch_embed1.num_patches * self.num_sa * block.expansion + 1, num_classes)

        self.add1 = nn.Linear(num_classes,256)
        self.relu = nn.ReLU(inplace=False) 

        self.add2 = nn.Linear(256,1)

        # self.add3 = nn.Linear(10,1)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    # 这里传入的参数block是Basci或者Bottleneck结构，blocks是创建Resnet的时候传入的[3，4，6，4]其中的一个具体的值，上面用layer[i]传入的；
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer_res
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 这里stride=2，因为上面传入的是2；所以这一层减少了fm的维度
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        # 这里stride=1，使用默认值；所以这里相当于没有做下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_gender):
        # See note [TorchScript super()]

        # 前面4层是静态的，所有Resnet的结构都要经过这四层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 后四层是动态的，也是Res18、Res34、Res50、Res101的区别
        x = self.layer1(x)  # 经过leyer1 ：w * H = 80 * 80
        y = x
        out1 = self.sa(y)

        x = self.layer2(x)  # 经过leyer2 ：w * H = 40 * 40
        y = x
        out2 = self.sa(y)
        

        x = self.layer3(x)  # 经过leyer3 ：w * H = 20 * 20
        y = x
        out3 = self.sa(y)
        

        x = self.layer4(x)  # 经过leyer4 ：W * H = 10 * 10
        y = x
        out4 = self.sa(y)
        

        # x = self.patch_embed(x)

        # x = self.blocks(x)

        B, C, _, _ = y.shape
        y = y.reshape(B,C,-1).permute(0,2,1)
        y = self.fc_y(y)
        y = y.permute(0,2,1)

        x = torch.concat((out1,out2,out3,out4),dim=1)
        # x = torch.concat((out3,out3,out4,out4),dim=1)
        # x = torch.concat((out4,out4,out4,out4),dim=1)


        # 用一个残差加起来
        x = x + y
        # x = y
        
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)

        # x = x.permute(1,0)

        x_gender = torch.unsqueeze(x_gender,1)
        x = torch.concat((x,x_gender),dim=-1)

        x = self.fc(x)

        x = self.add1(x)

        x = self.relu(x)

        x = self.add2(x)

        # x = self.relu(x)

        # x = self.add3(x)
        
        return x

    def forward(self, x, x_gender):
        return self._forward_impl(x, x_gender)



class Res_Vit_B16_gender_512_old(nn.Module):
    """
        Resnet最后的fc层参数改了;
        layer4和AveragePooling中间加了:PatchEmbed、Encoder Blocks * depth;

    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer_res=None, in_channels=3,\

                 img_size=10,patch_size=[16, 8, 4, 2],in_c=[64,128,256,512],embed_dim=EMBEDDING_DIM,depth=12,
                 num_heads=8,
                 mlp_ratio=4.0,qkv_bias=True,qk_scale=None,representation_size=None,
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,norm_layer_vit=None,act_layer=None,
                 num_sa = 4
                ):
        super(Res_Vit_B16_gender_512_old, self).__init__()
        
        if norm_layer_res is None:
            norm_layer_res = nn.BatchNorm2d
        self._norm_layer_res = norm_layer_res

        self.num_classed = num_classes
        norm_layer_vit = norm_layer_vit or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.inplanes = 64
        self.dilation = 1
        self.num_sa = num_sa
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer_res(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第一个layer不会下采样，因为stride默认为1；后面三个layer的stride都是2；
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.patch_embed1 = embed_layer(img_size=128,patch_size=patch_size[0],in_c=in_c[0],embed_dim=embed_dim)

        self.block1 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)
        


        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.patch_embed2 = embed_layer(img_size=64,patch_size=patch_size[1],in_c=in_c[1],embed_dim=embed_dim)

        self.block2 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)



        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.patch_embed3 = embed_layer(img_size=32,patch_size=patch_size[2],in_c=in_c[2],embed_dim=embed_dim)

        self.block3 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)



        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.patch_embed4 = embed_layer(img_size=16,patch_size=patch_size[3],in_c=in_c[3],embed_dim=embed_dim)

        self.block4 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)
        
        # self.attn = Attention(dim=embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
        #                       attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=drop_ratio)

        # self.patch_embed = embed_layer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            # 在一个列表里面用for循环创建depth(12)个Encoder_Block,之后再用nn.Sequential()方法全部打包起来
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], # ratio递增
                  norm_layer=norm_layer_vit, act_layer=act_layer)
            for i in range(depth)
        ])

        # 每层patch取256个时专用
        if patch_size[0] == 8:
            self.n256 = nn.Linear(256, 1024)

        elif patch_size[0] == 32:
            self.n256 = nn.Linear(256, 64)
        
        else:
            self.n256 = nn.Identity()



        self.fc_y = nn.Linear(in_c[3],768)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((self.patch_embed1.num_patches * self.num_sa, 1))

        self.fc = nn.Linear(self.patch_embed1.num_patches * self.num_sa + 1, num_classes)

        self.add1 = nn.Linear(num_classes,1024)
        # self.add1 = nn.Linear(num_classes,256)
        self.relu = nn.ReLU(inplace=False) 

        self.add2 = nn.Linear(1024,1)
        # self.add2 = nn.Linear(256,1)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    # 这里传入的参数block是Basci或者Bottleneck结构，blocks是创建Resnet的时候传入的[3，4，6，4]其中的一个具体的值，上面用layer[i]传入的；
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer_res
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 这里stride=2，因为上面传入的是2；所以这一层减少了fm的维度
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        # 这里stride=1，使用默认值；所以这里相当于没有做下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_gender):
        # See note [TorchScript super()]

        # 前面4层是静态的，所有Resnet的结构都要经过这四层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 这里写错了 y又经过了一次layer1 之前都是这么跑的!!!!
        x = self.layer1(x)  
        y = self.layer1(x)
        # y = x 按理说应该这么写
        out1 = self.patch_embed1(x)
        out1 = self.block1(out1)

        x = self.layer2(x)  
        y = self.layer2(y)
        out2 = self.patch_embed2(x)
        out2 = self.block2(out2)

        x = self.layer3(x)  
        y = self.layer3(y)
        out3 = self.patch_embed3(x)
        out3 = self.block3(out3)

        x = self.layer4(x)  
        y = self.layer4(y)
        out4 = self.patch_embed4(x)
        out4 = self.block4(out4)

        # x = self.patch_embed(x)

        # x = self.blocks(x)

        bs, c, _, _ = y.shape
        y = y.reshape(bs,c,-1).permute(0,2,1)
        y = self.fc_y(y)

        x = torch.concat((out1,out2,out3,out4),dim=1)
        # x = torch.concat((x,y),dim=2)
        # x = out1 + out2 + out3 + out4

        
        y = self.n256(y.transpose(-1, -2)).transpose(-1, -2)
        
        x = x + y
        
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)

        triplet_feature = x

        x_gender = torch.unsqueeze(x_gender,1)
        x = torch.concat((x,x_gender),dim=-1)

        x = self.fc(x)

        x = self.add1(x)

        x = self.relu(x)

        x = self.add2(x)
        

        return x, triplet_feature

    def forward(self, x, x_gender):
        return self._forward_impl(x, x_gender)



class Res_Vit_B16_gender_512_old_stn(nn.Module):
    """
        Resnet最后的fc层参数改了;
        layer4和AveragePooling中间加了:PatchEmbed、Encoder Blocks * depth;

    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer_res=None, in_channels=3,\

                 img_size=10,patch_size=[16, 8, 4, 2],in_c=[64,128,256,512],embed_dim=EMBEDDING_DIM,depth=12,
                 num_heads=8,
                 mlp_ratio=4.0,qkv_bias=True,qk_scale=None,representation_size=None,
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,norm_layer_vit=None,act_layer=None,
                 num_sa = 4
                ):
        super(Res_Vit_B16_gender_512_old_stn, self).__init__()
        
        if norm_layer_res is None:
            norm_layer_res = nn.BatchNorm2d
        self._norm_layer_res = norm_layer_res

        self.num_classed = num_classes
        norm_layer_vit = norm_layer_vit or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.inplanes = 64
        self.dilation = 1
        self.num_sa = num_sa
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer_res(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #----------------------------------------spatial transformers networks-------------------------------------
        self.localization = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(28 * 28 * 64, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            
        )

        # 第一个layer不会下采样，因为stride默认为1；后面三个layer的stride都是2；
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.patch_embed1 = embed_layer(img_size=128,patch_size=patch_size[0],in_c=in_c[0],embed_dim=embed_dim)

        self.block1 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)
        


        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.patch_embed2 = embed_layer(img_size=64,patch_size=patch_size[1],in_c=in_c[1],embed_dim=embed_dim)

        self.block2 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)



        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.patch_embed3 = embed_layer(img_size=32,patch_size=patch_size[2],in_c=in_c[2],embed_dim=embed_dim)

        self.block3 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)



        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.patch_embed4 = embed_layer(img_size=16,patch_size=patch_size[3],in_c=in_c[3],embed_dim=embed_dim)

        self.block4 = Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=0,
                            norm_layer=norm_layer_vit,act_layer=act_layer)
        
        # self.attn = Attention(dim=embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
        #                       attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=drop_ratio)

        # self.patch_embed = embed_layer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            # 在一个列表里面用for循环创建depth(12)个Encoder_Block,之后再用nn.Sequential()方法全部打包起来
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], # ratio递增
                  norm_layer=norm_layer_vit, act_layer=act_layer)
            for i in range(depth)
        ])

        # 每层patch取256个时专用
        if patch_size[0] == 8:
            self.n256 = nn.Linear(256, 1024)

        elif patch_size[0] == 32:
            self.n256 = nn.Linear(256, 64)
        
        else:
            self.n256 = nn.Identity()



        self.fc_y = nn.Linear(in_c[3],768)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((self.patch_embed1.num_patches * self.num_sa, 1))

        self.fc = nn.Linear(self.patch_embed1.num_patches * self.num_sa + 1, num_classes)

        self.add1 = nn.Linear(num_classes,1024)
        # self.add1 = nn.Linear(num_classes,256)
        self.relu = nn.ReLU(inplace=False) 

        self.add2 = nn.Linear(1024,1)
        # self.add2 = nn.Linear(256,1)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    # 这里传入的参数block是Basci或者Bottleneck结构，blocks是创建Resnet的时候传入的[3，4，6，4]其中的一个具体的值，上面用layer[i]传入的；
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer_res
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 这里stride=2，因为上面传入的是2；所以这一层减少了fm的维度
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        # 这里stride=1，使用默认值；所以这里相当于没有做下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_gender):
        # See note [TorchScript super()]

        # 前面4层是静态的，所有Resnet的结构都要经过这四层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        xs = self.localization(x)
        xs = xs.view(4, -1, 28 * 28 * 64)
        theta = self.fc_loc(xs)
        theta = theta.view(4, -1, 2, 3)

        theta = theta.squeeze(dim=1)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        # 这里写错了 y又经过了一次layer1 之前都是这么跑的!!!!
        x = self.layer1(x)  
        y = self.layer1(x)        
        out1 = self.patch_embed1(x)
        out1 = self.block1(out1)

        x = self.layer2(x)  
        y = self.layer2(y)
        out2 = self.patch_embed2(x)
        out2 = self.block2(out2)

        x = self.layer3(x)  
        y = self.layer3(y)
        out3 = self.patch_embed3(x)
        out3 = self.block3(out3)

        x = self.layer4(x)  
        y = self.layer4(y)
        out4 = self.patch_embed4(x)
        out4 = self.block4(out4)

        # x = self.patch_embed(x)

        # x = self.blocks(x)

        bs, c, _, _ = y.shape
        y = y.reshape(bs,c,-1).permute(0,2,1)
        y = self.fc_y(y)

        x = torch.concat((out1,out2,out3,out4),dim=1)
        # x = torch.concat((x,y),dim=2)
        # x = out1 + out2 + out3 + out4

        
        y = self.n256(y.transpose(-1, -2)).transpose(-1, -2)
        
        x = x + y
        
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)

        triplet_feature = x

        x_gender = torch.unsqueeze(x_gender,1)
        x = torch.concat((x,x_gender),dim=-1)

        x = self.fc(x)

        x = self.add1(x)

        x = self.relu(x)

        x = self.add2(x)
        

        return x, triplet_feature

    def forward(self, x, x_gender):
        return self._forward_impl(x, x_gender)





"""
SA:     不用patch_embedding的模型
n64:    每个stage取64个patch
n256:   每个stage取256个patch
old:    之前layer1经过了两次的模型

"""


def Res18_LCFF_V1():
    model = Res_Vit_B16_gender_512_new(block=BasicBlock, layers=[2,2,2,2], in_channels=3, depth=0, embed_dim=768, drop_path_ratio=0)
    return model

def Res34_LCFF_V1():
    model = Res_Vit_B16_gender_512_new(block=BasicBlock, layers=[3,4,6,3], in_channels=3, depth=0, embed_dim=768,)
    return model

def Res50_LCFF_V1_n64():
    model = Res_Vit_B16_gender_512_new(block=Bottleneck, layers=[3,4,6,3], in_channels=3, depth=0, embed_dim=768,in_c=[256,512,1024,2048], drop_path_ratio=0.2)
    return model

def Res101_LCFF_V1():
    model = Res_Vit_B16_gender_512_new(block=Bottleneck, layers=[3,4,23,3], in_channels=3, depth=0, embed_dim=768,in_c=[256,512,1024,2048])
    return model

def Res50_LCFF_V1_SA():
    # 最上面的EMBEDDING_DIM改成256才行
    model = Res_Vit_B16_gender_512_sa1_bottleneck(block=Bottleneck, layers=[3,4,6,3], in_channels=3, embedding_dim=256)
    return model

def Res34_LCFF_V1_old():
    model = Res_Vit_B16_gender_512_old(block=BasicBlock, layers=[3,4,6,3], in_channels=3, depth=0, embed_dim=768,)
    return model

def Res34_LCFF_V1_n256():
    model = Res_Vit_B16_gender_512_old(block=BasicBlock, layers=[3,4,6,3], in_channels=3, depth=0, embed_dim=768, patch_size=[8,4,2,1])
    return model

def Res34_LCFF_V1_n16():
    model = Res_Vit_B16_gender_512_old(block=BasicBlock, layers=[3,4,6,3], in_channels=3, depth=0, embed_dim=768, patch_size=[32,16,8,4])
    return model

def Res34_LCFF_V1_old_stn():
    model = Res_Vit_B16_gender_512_old_stn(block=BasicBlock, layers=[3,4,6,3], in_channels=3, depth=0, embed_dim=768,)
    return model

