import gzip
import pickle
import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import cv2
import torch.nn.functional as F


# data augmentation
PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)

def _load_data(url, filename):
    """Load data from `url` and store the result in `filename`."""
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')



def load_data(filename, url=None):
    """Get data with labels, split into training, validation and test set."""
    data = _load_data(url,filename)
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=10)


ORG_SHP = [28,28]
OUT_SHP = [100,100]
NUM_DISTORTIONS = 6
dist_size = (9,9)  
NUM_DISTORTIONS_DB = 100000

mnist_data = load_data('data/mnist.pkl.gz')
np.random.seed(1234)
''' mnist dataset mnist.pkl.gz
contains: X_train (50000),X_vaild (10000),X_test (10000), each img of size 784
input dim:784
output_dim:(10)
'''
### create list with distortions
all_digits = np.concatenate([mnist_data['X_train'], mnist_data['X_valid']], axis=0)
all_digits = all_digits.reshape([-1] + ORG_SHP) #(600000,28,28)
num_digits = all_digits.shape[0] 

distortions = []
for i in range(NUM_DISTORTIONS_DB):
    rand_digit = np.random.randint(num_digits)
    rand_x = np.random.randint(ORG_SHP[1]-dist_size[1])
    rand_y = np.random.randint(ORG_SHP[0]-dist_size[0])

    digit = all_digits[rand_digit]
    distortion = digit[rand_y:rand_y + dist_size[0],
                       rand_x:rand_x + dist_size[1]]
    assert distortion.shape == dist_size
    distortions += [distortion]
print("Created distortions")
def create_sample1(x, output_shp, num_distortions=NUM_DISTORTIONS):
    a, b= x.shape
    x_offset = (output_shp[1]-a)//2


    #x_offset += np.random.choice(range(-x_offset, x_offset))
    y_offset = (output_shp[1]-a)//2


    angle = np.random.choice(range(int(-b*0.5), int(b*0.5)))

    output = np.zeros(output_shp)
    
    x_start = 0*b+x_offset

    x_end = x_start + b
    y_start = y_offset + np.floor(0*angle)
    y_end = y_start + a
    if y_end > (output_shp[1]-1):
        m = output_shp[1] - y_end
        y_end += m
        y_start += m
    if y_start < 0:
        m = y_start
        y_end -= m
        y_start -= m
    y_start,y_end=int(y_start),int(y_end)
    
    output[y_start:y_end, x_start:x_end] = x

    if num_distortions > 0:
            output = add_distortions(output, num_distortions)
    return output



def add_distortions(digits, num_distortions):
    canvas = np.zeros_like(digits)
    for i in range(num_distortions):
        rand_distortion = distortions[np.random.randint(NUM_DISTORTIONS_DB)]
        rand_x = np.random.randint(OUT_SHP[1]-dist_size[1])
        rand_y = np.random.randint(OUT_SHP[0]-dist_size[0])
        canvas[rand_y:rand_y+dist_size[0],
               rand_x:rand_x+dist_size[1]] = rand_distortion
    canvas += digits

    return np.clip(canvas, 0, 1)


class DataAugmentation:
    def __init__(self,global_crops_scale=(0.5,1.0),n_local_crops=2,output_size=224):
        
        self.n_local_crops = n_local_crops
        RandomGaussianBlur=lambda p: transforms.RandomApply([transforms.GaussianBlur(kernel_size=3,sigma=(0.1,2))],p=p)
        flip_and_rotation=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomRotation(degrees=(10)),])
        colorjitter=transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.2)
        crop=transforms.CenterCrop(28)
        resize=transforms.Resize((output_size,output_size),interpolation=InterpolationMode.BICUBIC)
        rotation=transforms.RandomRotation(degrees=(6))
        shift=transforms.RandomAffine(degrees=6,translate=(0.2,0.1))
        normalize=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.13,),(0.3,)),])
        

        self.global_1=transforms.Compose([
            #shift,
            #flip_and_rotation,
            transforms.RandomResizedCrop(output_size,scale=global_crops_scale,interpolation=InterpolationMode.BICUBIC),
            #colorjitter,
            rotation,
            RandomGaussianBlur(0.1),
            normalize
        ])
        self.global_2=transforms.Compose([
            transforms.RandomResizedCrop(output_size,scale=global_crops_scale,interpolation=InterpolationMode.BICUBIC),
            rotation,
            #colorjitter,
            RandomGaussianBlur(1.0),
            #transforms.RandomSolarize(170,p=0.2),
            normalize
        ])
        self.local=transforms.Compose([
            crop,
            resize,
            #colorjitter,
            rotation,
            RandomGaussianBlur(0.5),
            normalize
        ])

    
    def __call__(self,image):
        '''
        all_crops:list of torch.Tensor
        represent different version of input img
        '''
        all_crops=[]
        image=(np.asarray(image.convert('L')))/255.0
        
        image1=create_sample1(image, OUT_SHP)
        image2=create_sample1(image, OUT_SHP)
        image1=(image1*255.0).astype(np.uint8)
        image2=(image2*255.0).astype(np.uint8)
        image1=Image.fromarray(cv2.cvtColor(image1,cv2.COLOR_GRAY2RGB))
        image2=Image.fromarray(cv2.cvtColor(image2,cv2.COLOR_GRAY2RGB))

        all_crops.append(self.global_1(image1))
        all_crops.append(self.global_2(image1))
        all_crops.append(self.local(image1))
        all_crops.append(self.local(image1))
        

        all_crops.append(self.global_1(image2))
        all_crops.append(self.global_2(image2))
        all_crops.append(self.local(image2))
        all_crops.append(self.local(image2))
        return all_crops

# Vision Transformer
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import math
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


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
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bias=False,qk_scale=None,attn_drop_ratio=0.,proj_drop_ratio=0.):
        super(Attention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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


class Block(nn.Module):
    def __init__(self,dim,num_heads, mlp_ratio=4.,qkv_bias=False,qk_scale=None, drop_ratio=0.,attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        #  drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x,return_attention=False):
        y,attn=self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_c=3, num_classes=0,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Parameters:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit_weights)
    
    
    def _init_vit_weights(self,m):
        """
        ViT weight initialization
        :param m: module
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

#     def forward_features(self, x):
#         # [B, C, H, W] -> [B, num_patches, embed_dim]
#         x = self.patch_embed(x)  # [B, 196, 768]
#         # [1, 1, 768] -> [B, 1, 768]
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         if self.dist_token is None:
#             x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
#         else:
#             x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

#         x = self.pos_drop(x + self.pos_embed)
#         x = self.blocks(x)
#         x = self.norm(x)
#         if self.dist_token is None:
#             return self.pre_logits(x[:, 0])
#         else:
#             return x[:, 0], x[:, 1]

#     def forward(self, x):
#         x = self.forward_features(x)
#         if self.head_dist is not None:
#             x, x_dist = self.head(x[0]), self.head_dist(x[1])
#             if self.training and not torch.jit.is_scripting():
#                 # during inference, return the average of both classifier predictions
#                 return x, x_dist
#             else:
#                 return (x + x_dist) / 2
#         else:
#             x = self.head(x)
#         return x
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)
            
    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


#new head
class DINOHead(nn.Module):
    """Network hooked up to the CLS token embedding.
    Just a MLP with the last layer being normalized in a particular way.
    
    Parameters:
    in_dim : int
        The dimensionality of the token embedding.
    out_dim : int
        The dimensionality of the final layer (we compute the softmax over).
    hidden_dim : int
        Dimensionality of the hidden layers.
    bottleneck_dim : int
        Dimensionality of the second last layer.
    n_layers : int
        The number of layers.
    norm_last_layer : bool
        If True, then we freeze the norm of the weight of the last linear layer
        to 1.
        
        
    Attributes:
    mlp : nn.Sequential
        Vanilla multi-layer perceptron.
    last_layer : nn.Linear
        Reparametrized linear layer with weight normalization. That means
        that that it will have `weight_g` and `weight_v` as learnable
        parameters instead of a single `weight`.
    """
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=512, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass.
        
        Parameters:
        x : torch.Tensor
            Of shape `(n_samples, in_dim)`.
        
        return: torch.Tensor
            Of shape `(n_samples, out_dim)`.
        """
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


#Multicropwrapper
class MultiCropWrapper(nn.Module):
    """Convenience class for forward pass of multiple crops.

    Parameters:
    backbone : vision transformer
        Instantiated Vision Transformer. Note that we will take the `head` attribute and replace it with `nn.Identity`.
    head : DINOHead
        New head that is going to be put on top of the `backbone`.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        '''
        The different crops are concatenated along the batch dimension. The resulting tensor is then chunked back to per crop tensors.
        return: list of crops len=n_crops, each of shape (batch,out_dim)
        '''
        # convert to list
        if not isinstance(x, list):
            #print('multicrop',x.shape)
            x = [x]
        n_crops=len(x)
        concatenated=torch.cat(x,dim=0)
        cls_embedding=self.backbone(concatenated)
        logits=self.head(cls_embedding)
        chunks=logits.chunk(n_crops)
        return chunks

#simCLR loss

class Loss(nn.Module):
    def __init__(self, tau=0.1,out_dim=1024,center_momentum=0.995):
        super().__init__()
        self.tau=tau
        self.center_momentum=center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output_ori, teacher_output_ori):
        """
        simCLRLoss of the teacher and student branch.
        student_output_ori: list of len=n_crops, each of shape (batch,out_dim)
        """
        teacher_output=torch.cat(teacher_output_ori,dim=1)
        student_output=torch.cat(student_output_ori,dim=1)
        n_examples,_=student_output.size()
        teacher=F.normalize(teacher_output,dim=-1)
        student=F.normalize(student_output,dim=-1)
        scores=torch.mm(teacher,student.t()).div_(self.tau)
        target=torch.arange(n_examples,dtype=torch.long).to(scores.device)
        loss=F.cross_entropy(scores,target)
        self.update_center(teacher_output_ori)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.
        Compute the exponential moving average.
        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

    
def clip_gradients(model, clip=2.0):
    """Rescale norm of computed gradients. Used to avoid gradient exponential
    Parameters
    ----------
    model : nn.Module
        Module.
    clip : float
        Maximum norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)