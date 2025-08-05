import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sam2.build_sam import build_sam2
from timm.models.layers import trunc_normal_   


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)

    
class SAM2UNeXT(nn.Module):
    def __init__(self, checkpoint_path=None, dinov2_path=None) -> None:
        super(SAM2UNeXT, self).__init__()

        # ===== SAM2 Encoder =====    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.sam = model.image_encoder.trunk
        for param in self.sam.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.sam.blocks:
            blocks.append(
                Adapter(block)
            )
        self.sam.blocks = nn.Sequential(
            *blocks
        )

        # ===== DINOv2 Encoder =====
        if dinov2_path:
            self.dino = timm.create_model('vit_large_patch14_dinov2',
                                        features_only=True,
                                        img_size=(448, 448),
                                        pretrained=True,
                                        pretrained_cfg_overlay=dict(file=dinov2_path))
        else:
            self.dino = timm.create_model('vit_large_patch14_dinov2',
                                        features_only=True,
                                        img_size=(448, 448))
        for param in self.dino.parameters():
            param.requires_grad = False

        self.align1 = nn.Conv2d(1024, 144, 1)
        self.align2 = nn.Conv2d(1024, 288, 1)
        self.align3 = nn.Conv2d(1024, 576, 1)
        self.align4 = nn.Conv2d(1024, 1152, 1)

        self.reduce1 = nn.Conv2d(144+144, 128, 1)
        self.reduce2 = nn.Conv2d(288+288, 128, 1)
        self.reduce3 = nn.Conv2d(576+576, 128, 1)
        self.reduce4 = nn.Conv2d(1152+1152, 128, 1)

        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)
        

    def forward(self, x):
        x1_s, x2_s, x3_s, x4_s = self.sam(x)
        x_d = self.dino(F.interpolate(x, size=(448, 448), mode='bilinear'))[-1]

        x1_d = F.interpolate(self.align1(x_d), size=x1_s.shape[-2:], mode='bilinear')
        x2_d = F.interpolate(self.align2(x_d), size=x2_s.shape[-2:], mode='bilinear')
        x3_d = F.interpolate(self.align3(x_d), size=x3_s.shape[-2:], mode='bilinear')
        x4_d = F.interpolate(self.align4(x_d), size=x4_s.shape[-2:], mode='bilinear')

        x1, x2, x3, x4 = torch.cat([x1_s,x1_d], dim=1), torch.cat([x2_s,x2_d], dim=1), torch.cat([x3_s,x3_d], dim=1), torch.cat([x4_s,x4_d], dim=1)
        x1, x2, x3, x4 = self.reduce1(x1), self.reduce2(x2), self.reduce3(x3), self.reduce4(x4)
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        out = F.interpolate(self.head(x), scale_factor=2, mode='bilinear')
        return out
