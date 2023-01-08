import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


# patch_size = 16
# patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

class PatchEmbedding(nn.Module):
    #def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
    #def __init__(self, in_channels: int = 3, patch_size: int = 8, emb_size: int = 192, img_size: int = 32):
    def __init__(self, in_channels: int = 1, patch_size: int = 7, emb_size: int = 49, img_size: int = 28):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 이미지를 s1 x s2 패치로 쪼개고 flatten
            # 8×3×(14∗16)×(14∗16) 을 8×(14∗14)×(16∗16∗3) 형태로 바꾸어줌
            # 3 * (4*8) * 4*8) -> (4*4)*(8*8*3)
            # batch_size (b) = 8, patch_size (s1, s2) = 16, channels_size (c) = 3
            # (w, h) = 14, 14
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        # cls_token 정의
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # position Parameter 정의
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        # cls_token을 반복해, 배치 사이즈와 크기 맞춰줌
        # (1, 1, 768) -> (8, 1, 768)
        # b -> batch_size = 8
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # cls_token과 projection을 concatenate
        x = torch.cat([cls_tokens, x], dim=1)
        # position encoding을 더해 줌
        # (197, 768)
        # (8, 197, 768)
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    # def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.0):
    # def __init__(self, emb_size: int = 192, num_heads: int = 8, dropout: float = 0.0):
    def __init__(self, emb_size: int = 49, num_heads: int = 7, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        # QKV
        # self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(p=dropout)

        #Linear Projection
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 8 197 (8 96 3) -> (3) 8 8 197 96
        # qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        # queries = qkv[0]
        # keys = qkv[1]
        # values = qkv[2]

        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # sum
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32)
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=1) / scaling
        att = self.att_drop(att)

        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 # emb_size=768,
                 # emb_size=192,
                 emb_size=49,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                ),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassficationHead(nn.Sequential):
    # def __init__(self, emb_size=768, n_classes=1000):
    # def __init__(self, emb_size=192, n_classes=10):
    def __init__(self, emb_size=49, n_classes=10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

class ViT(nn.Sequential):
    def __init__(self,
                 # in_channels: int = 3,
                 # patch_size: int = 16,
                 # emb_size: int = 768,
                 # img_size: int = 224,
                 # patch_size: int = 8,
                 # emb_size: int = 192,
                 # img_size: int = 32,
                 in_channels: int = 1,
                 patch_size: int = 7,
                 emb_size: int = 49,
                 img_size: int = 28,
                 depth: int = 12,
                 # n_classes: int = 1000,
                 n_classes: int = 10,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassficationHead(emb_size, n_classes)
        )

# summary(ViT(), (1, 28, 28), device='cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x