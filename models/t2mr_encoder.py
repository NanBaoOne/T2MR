import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import import_class
from models.ms_gcn import MultiScale_GraphConv as MS_GCN
from models.ms_tcn import MultiScale_TemporalConv as MS_TCN
from models.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from models.mlp import MLP


class T2MRBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )
        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        x = self.gcn3d(x)
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)
        return x


class MultiWindowT2MRBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3, 5],
                 window_stride=1,
                 window_dilations=[1, 1]):
        super().__init__()
        self.gcn3d = nn.ModuleList([
            T2MRBlock(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        return out_sum


class T2MRMotionEncoder(nn.Module):
    """
    T²MR motion encoder for NTU 25-joint skeleton.
    Compatible with the pre-training checkpoint used in this project.
    """

    def __init__(self,
                 num_point=25,
                 in_channels=3,
                 embed_dim=384,
                 num_gcn_scales=13,
                 num_g3d_scales=6,
                 graph='graph.smpl_graph.Graph'):
        super().__init__()

        # 使用NTU图结构（25关节）
        Graph = import_class(graph)
        A_binary = Graph().A_binary

        # BatchNorm: 25关节 * 3维度 = 75
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)

        c1, c2, c3 = 96, 192, embed_dim

        # Block 1: stride=1
        self.gcn3d1 = MultiWindowT2MRBlock(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1)
        )
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        # Block 2: stride=2
        self.gcn3d2 = MultiWindowT2MRBlock(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2)
        )
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        # Block 3: stride=2
        self.gcn3d3 = MultiWindowT2MRBlock(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3)
        )
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.out_channels = c3

    def forward(self, x):
        """
        输入: x shape (B, T, V, C) = (B, seqlen, 25, 3)
        输出: shape (B, embed_dim)
        """
        B, T, V, C = x.shape

        # Convert to the internal tensor layout used by this motion encoder.
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, 3, T, 25, 1)

        N, C_in, T_in, V_in, M = x.shape

        # BatchNorm
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_in * C_in, T_in)
        x = self.data_bn(x)
        x = x.view(N * M, V_in, C_in, T_in).permute(0, 2, 3, 1).contiguous()

        # Block 1
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        # Block 2
        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)

        # Block 3
        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        # Global Average Pooling
        x = x.mean(dim=[2, 3])  # (B, embed_dim)

        return x

    def load_pretrained(self, ckpt_path, device='cpu', strict=True):
        """
        加载预训练权重
        由于使用相同的25关节NTU结构，可以完全加载（包括邻接矩阵）

        Args:
            ckpt_path: 预训练模型路径
            device: 设备
            strict: 是否严格匹配（推荐True，因为结构完全一致）
        """
        print(f"\n{'=' * 60}")
        print(f"Loading pretrained weights from: {ckpt_path}")
        print(f"{'=' * 60}")

        checkpoint = torch.load(ckpt_path, map_location=device)

        # 提取state_dict
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        model_dict = self.state_dict()

        # 过滤掉分类头（fc层）
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and not k.startswith('fc.')
        }

        # 检查形状匹配
        matched = {}
        skipped = []

        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    matched[k] = v
                else:
                    skipped.append((k, v.shape, model_dict[k].shape))
            else:
                skipped.append((k, v.shape, None))

        # 加载权重
        model_dict.update(matched)
        self.load_state_dict(model_dict, strict=False)

        # 打印统计
        print(f"\n{'=' * 60}")
        print(f"Loading Summary:")
        print(f"{'=' * 60}")
        print(f"✓ Matched layers:      {len(matched)}/{len(model_dict)}")
        print(f"⊘ Skipped layers:      {len(skipped)}")

        if skipped:
            print(f"\nSkipped layers:")
            for item in skipped[:5]:
                if len(item) == 3:
                    name, pre_shape, cur_shape = item
                    if cur_shape is None:
                        print(f"  - {name}: not in current model")
                    else:
                        print(f"  - {name}: shape {pre_shape} vs {cur_shape}")
            if len(skipped) > 5:
                print(f"  ... and {len(skipped) - 5} more")

        load_ratio = len(matched) / len(model_dict)
        print(f"\n✓ Successfully loaded {load_ratio:.1%} of model parameters")
        print(f"{'=' * 60}\n")

        return len(matched), len(skipped)