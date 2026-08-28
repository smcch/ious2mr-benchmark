"""
3D variants of the NCSNpp generator and Discriminator_large used by the
paired SynDiff trainer. Self-contained: does NOT depend on the FIR
upfirdn2d CUDA op (no 3D version exists), so we fall back to trilinear
upsample + AvgPool3d downsample as in DDPM-style nets.

DESIGN CHOICE: Option A from the spec — write a fresh `NCSNpp3D` with
pure Conv3d / GroupNorm3d-friendly layers. Option B (monkey-patching the
2D code) was rejected because layerspp internally uses
`up_or_down_sampling.upsample_2d` (CUDA-only, FIR-based) and there is no
clean drop-in for that operator in 3D.

Memory model (RTX 4080 Super, 16 GB):
  - patch_size = 64, ngf = num_channels_dae = 24, ch_mult = [1,2,2,2]
    -> 4 levels (64 -> 32 -> 16 -> 8). Original 6-level [1,1,2,2,4,4]
    would put activations of (B, 96, 16, 16, 16) at the bottleneck which
    exceeds the budget; 4 levels keep peak resolution-channel product
    well under the 12 GB target.
  - attn_resolutions = (8,) — attention quadratic in spatial volume; in
    3D a 16^3 attn map is 4096^2 entries per head per batch, OOM. 8^3
    is 512^2 which fits.
  - We expose `forward_block_ckpt` so the trainer can wrap each ResNet
    block with `torch.utils.checkpoint.checkpoint` (saves activations
    by recomputing on backward; trades ~30% extra compute for ~3x less
    activation memory).
  - Output is single channel (B, 1, D, H, W) to match the 2D NCSNpp's
    single-channel output convention.

The config attribute names (image_size, num_channels, num_channels_dae,
ch_mult, num_res_blocks, attn_resolutions, dropout, resamp_with_conv,
conditional, embedding_type, fourier_scale, not_use_tanh, centered,
n_mlp, nz, z_emb_dim, t_emb_dim, ngf, t_emb_dim) mirror the 2D NCSNpp
exactly so the trainer can reuse the same `args`-style config object.
"""
from __future__ import annotations
import math
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Init helpers (mirror layers.default_init / ddpm_conv3x3)
# --------------------------------------------------------------------------- #
def _variance_scaling(scale, mode="fan_avg", distribution="uniform"):
    def init(shape):
        # compute fans (treating last two indices as kernel)
        receptive = 1
        for s in shape[2:]:
            receptive *= s
        fan_in = shape[1] * receptive
        fan_out = shape[0] * receptive
        if mode == "fan_in":
            denom = fan_in
        elif mode == "fan_out":
            denom = fan_out
        else:
            denom = (fan_in + fan_out) / 2
        var = scale / max(denom, 1)
        if distribution == "uniform":
            return (torch.rand(*shape) * 2 - 1) * math.sqrt(3 * var)
        else:
            return torch.randn(*shape) * math.sqrt(var)
    return init


def default_init3d(scale=1.0):
    scale = 1e-10 if scale == 0 else scale
    return _variance_scaling(scale, "fan_avg", "uniform")


def conv3x3x3(in_ch, out_ch, stride=1, padding=1, bias=True, init_scale=1.0):
    c = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride,
                  padding=padding, bias=bias)
    c.weight.data = default_init3d(init_scale)(c.weight.shape)
    if bias:
        nn.init.zeros_(c.bias)
    return c


def conv1x1x1(in_ch, out_ch, stride=1, bias=True, init_scale=1.0):
    c = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=bias)
    c.weight.data = default_init3d(init_scale)(c.weight.shape)
    if bias:
        nn.init.zeros_(c.bias)
    return c


def dense_init(in_dim, out_dim, init_scale=1.0):
    lin = nn.Linear(in_dim, out_dim)
    # 2-D weight; reuse the same fan_avg uniform init
    fan_in, fan_out = in_dim, out_dim
    var = (1e-10 if init_scale == 0 else init_scale) / ((fan_in + fan_out) / 2)
    lin.weight.data = (torch.rand(out_dim, in_dim) * 2 - 1) * math.sqrt(3 * var)
    nn.init.zeros_(lin.bias)
    return lin


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """Sinusoidal pos enc — identical to syndiff_src/backbones/layers.py."""
    assert timesteps.dim() == 1
    half = embedding_dim // 2
    freqs = math.log(max_positions) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32,
                                    device=timesteps.device) * -freqs)
    emb = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# --------------------------------------------------------------------------- #
# 2. Building blocks
# --------------------------------------------------------------------------- #
class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class AdaptiveGroupNorm3d(nn.Module):
    """AdaIN-style group norm modulated by the z-embedding (style)."""
    def __init__(self, num_groups, in_channel, style_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense_init(style_dim, in_channel * 2)
        # match 2D init
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, x, style):
        # style : (B, style_dim)
        s = self.style(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        gamma, beta = s.chunk(2, dim=1)
        return gamma * self.norm(x) + beta


class Upsample3d(nn.Module):
    """Trilinear up by 2; optional 3x3x3 conv. NO FIR (no CUDA op exists)."""
    def __init__(self, in_ch, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = conv3x3x3(in_ch, in_ch)

    def forward(self, x):
        # trilinear is the cheapest 3D up-op that supports backprop;
        # nearest is faster but causes block artefacts in 3D.
        h = F.interpolate(x, scale_factor=2, mode="trilinear",
                          align_corners=False)
        if self.with_conv:
            h = self.conv(h)
        return h


class Downsample3d(nn.Module):
    """AvgPool3d by 2 (or strided 3x3x3 conv if with_conv)."""
    def __init__(self, in_ch, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = conv3x3x3(in_ch, in_ch, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        return F.avg_pool3d(x, 2)


class AttnBlock3d(nn.Module):
    """Self-attention over the 3D spatial volume.
    Memory: O((D*H*W)^2) attention map per head. Use ONLY at low res
    (resolution=8 in our config -> 512 tokens, fine)."""
    def __init__(self, channels, init_scale=0.0, skip_rescale=True):
        super().__init__()
        groups = min(channels // 4, 32)
        if groups < 1: groups = 1
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)
        self.q = conv1x1x1(channels, channels)
        self.k = conv1x1x1(channels, channels)
        self.v = conv1x1x1(channels, channels)
        self.proj = conv1x1x1(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale
        self.channels = channels

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.gn(x)
        q = self.q(h).reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        k = self.k(h).reshape(B, C, -1)                    # (B, C, N)
        v = self.v(h).reshape(B, C, -1).permute(0, 2, 1)   # (B, N, C)
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, D, H, W)
        out = self.proj(out)
        if self.skip_rescale:
            return (x + out) / math.sqrt(2.0)
        return x + out


class ResnetBlockBigGAN3d(nn.Module):
    """3D BigGAN-style residual block with adagn (z-emb) + temb bias.

    Mirrors layerspp.ResnetBlockBigGANpp_Adagn:
        h = act(adagn(norm(x), zemb))
        h = up/down(h)
        h = conv0(h)
        h += dense(act(temb))
        h = act(adagn(norm(h), zemb))
        h = drop(h)
        h = conv1(h)  (init_scale=0)
        skip = up/down(x); if dim mismatch -> 1x1 conv
        return (h + skip) / sqrt(2)
    """
    def __init__(self, act, in_ch, out_ch=None, *, temb_dim=None, zemb_dim=None,
                 up=False, down=False, dropout=0.0, init_scale=0.0,
                 skip_rescale=True, with_resample_conv=True):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.up = up
        self.down = down
        self.skip_rescale = skip_rescale
        self.act = act

        groups_in = min(in_ch // 4, 32);  groups_in = max(groups_in, 1)
        groups_out = min(out_ch // 4, 32); groups_out = max(groups_out, 1)

        self.gn0 = nn.GroupNorm(num_groups=groups_in, num_channels=in_ch, eps=1e-6)
        self.adagn0 = AdaptiveGroupNorm3d(groups_in, in_ch, zemb_dim) if zemb_dim else None

        self.conv0 = conv3x3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.temb_proj = dense_init(temb_dim, out_ch)
        else:
            self.temb_proj = None

        self.gn1 = nn.GroupNorm(num_groups=groups_out, num_channels=out_ch, eps=1e-6)
        self.adagn1 = AdaptiveGroupNorm3d(groups_out, out_ch, zemb_dim) if zemb_dim else None
        self.dropout = nn.Dropout(dropout)
        self.conv1 = conv3x3x3(out_ch, out_ch, init_scale=init_scale)

        if up or down or in_ch != out_ch:
            self.shortcut = conv1x1x1(in_ch, out_ch)
        else:
            self.shortcut = None

        # resample modules (no FIR — trilinear up / avgpool down)
        self._up = Upsample3d(in_ch, with_conv=False) if up else None
        self._down = Downsample3d(in_ch, with_conv=False) if down else None

    def _resample(self, x):
        if self._up is not None: return self._up(x)
        if self._down is not None: return self._down(x)
        return x

    def forward(self, x, temb, zemb):
        h = self.gn0(x)
        if self.adagn0 is not None and zemb is not None:
            h = self.adagn0(h, zemb)
        h = self.act(h)

        # Resample BEFORE the first conv (BigGAN convention)
        if self.up or self.down:
            h = self._resample(h)
            x = self._resample(x)

        h = self.conv0(h)
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(self.act(temb))[:, :, None, None, None]

        h = self.gn1(h)
        if self.adagn1 is not None and zemb is not None:
            h = self.adagn1(h, zemb)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv1(h)

        if self.shortcut is not None:
            x = self.shortcut(x)

        if self.skip_rescale:
            return (x + h) / math.sqrt(2.0)
        return x + h


# --------------------------------------------------------------------------- #
# 3. NCSNpp3D — the generator
# --------------------------------------------------------------------------- #
class NCSNpp3D(nn.Module):
    """3D NCSNpp generator. Configuration mirrors the 2D NCSNpp; only
    spatially-3D variants are used (Conv3d, GroupNorm, AvgPool3d, trilinear
    upsample). Output is single-channel (B, 1, D, H, W)."""

    def __init__(self, config, use_grad_ckpt: bool = True):
        super().__init__()
        self.config = config
        self.use_grad_ckpt = use_grad_ckpt
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        nf = config.num_channels_dae
        self.nf = nf
        self.ch_mult = list(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.attn_resolutions = tuple(config.attn_resolutions)
        dropout = config.dropout
        self.num_resolutions = len(self.ch_mult)
        # spatial size at each level (cubic patch, only one dim tracked)
        self.all_resolutions = [config.image_size // (2 ** i)
                                for i in range(self.num_resolutions)]
        self.conditional = config.conditional
        self.embedding_type = config.embedding_type.lower()
        self.skip_rescale = getattr(config, "skip_rescale", True)
        init_scale = 0.0
        in_channels = config.num_channels  # 2 (US + noisy T2 channels concat'd)

        # ---- time embedding head ----
        if self.embedding_type == "fourier":
            # Gaussian Fourier features
            scale = float(getattr(config, "fourier_scale", 16.0))
            self.fourier_W = nn.Parameter(
                torch.randn(nf) * scale, requires_grad=False)
            embed_dim = 2 * nf
        else:
            embed_dim = nf  # positional sinusoid

        if self.conditional:
            self.time_mlp = nn.Sequential(
                dense_init(embed_dim, nf * 4),
                act,
                dense_init(nf * 4, nf * 4),
            )
        else:
            self.time_mlp = None

        # ---- z mapping (PixelNorm + n_mlp dense) ----
        z_layers = [PixelNorm(), dense_init(config.nz, z_emb_dim), act]
        for _ in range(config.n_mlp):
            z_layers += [dense_init(z_emb_dim, z_emb_dim), act]
        self.z_transform = nn.Sequential(*z_layers)

        # ---- ResNet block factory ----
        ResBlock = functools.partial(
            ResnetBlockBigGAN3d,
            act=act,
            dropout=dropout,
            temb_dim=nf * 4,
            zemb_dim=z_emb_dim,
            init_scale=init_scale,
            skip_rescale=self.skip_rescale,
        )

        # ---- Encoder ----
        self.input_conv = conv3x3x3(in_channels, nf)
        hs_c = [nf]
        in_ch = nf

        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()       # parallel list (None where no attn)
        self.down_resamples = nn.ModuleList()   # one per level (except last)

        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                out_ch = nf * self.ch_mult[i_level]
                self.down_blocks.append(ResBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if self.all_resolutions[i_level] in self.attn_resolutions:
                    self.down_attns.append(AttnBlock3d(in_ch))
                else:
                    self.down_attns.append(None)
                hs_c.append(in_ch)

            if i_level != self.num_resolutions - 1:
                self.down_resamples.append(ResBlock(in_ch=in_ch, down=True))
                hs_c.append(in_ch)

        self.hs_c = hs_c

        # ---- Bottleneck ----
        self.mid_block1 = ResBlock(in_ch=in_ch)
        self.mid_attn = AttnBlock3d(in_ch)
        self.mid_block2 = ResBlock(in_ch=in_ch)

        # ---- Decoder ----
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_resamples = nn.ModuleList()

        # rebuild a mock hs_c stack to compute concat sizes during construction
        rev_hs = list(hs_c)
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                out_ch = nf * self.ch_mult[i_level]
                skip_ch = rev_hs.pop()
                self.up_blocks.append(ResBlock(in_ch=in_ch + skip_ch, out_ch=out_ch))
                in_ch = out_ch
            if self.all_resolutions[i_level] in self.attn_resolutions:
                self.up_attns.append(AttnBlock3d(in_ch))
            else:
                self.up_attns.append(None)
            if i_level != 0:
                self.up_resamples.append(ResBlock(in_ch=in_ch, up=True))
        assert not rev_hs, f"unconsumed skip channels: {rev_hs}"

        # ---- Output head ----
        out_groups = min(in_ch // 4, 32); out_groups = max(out_groups, 1)
        self.final_gn = nn.GroupNorm(num_groups=out_groups, num_channels=in_ch,
                                     eps=1e-6)
        # ALWAYS output 1 channel to match the 2D variant (gen returns target T2)
        self.final_conv = conv3x3x3(in_ch, 1, init_scale=init_scale)

    # ----- helpers --------------------------------------------------------- #
    def _ckpt_block(self, block, *args):
        """Apply gradient checkpointing if enabled and we're in training mode."""
        if self.use_grad_ckpt and self.training and torch.is_grad_enabled():
            # use_reentrant=False keeps it well-behaved with autocast bf16
            return torch.utils.checkpoint.checkpoint(
                block, *args, use_reentrant=False)
        return block(*args)

    def _embed_time(self, time_cond):
        if self.embedding_type == "fourier":
            x_proj = (torch.log(time_cond)[:, None] * self.fourier_W[None, :]
                      * 2 * math.pi)
            temb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        else:
            temb = get_timestep_embedding(time_cond, self.nf)
        if self.conditional:
            temb = self.time_mlp(temb)
        return temb

    # ----- forward --------------------------------------------------------- #
    def forward(self, x, time_cond, z):
        zemb = self.z_transform(z)
        temb = self._embed_time(time_cond)

        if not self.config.centered:
            x = 2.0 * x - 1.0

        # ---- encoder ----
        hs = [self.input_conv(x)]
        idx_b = idx_a = idx_r = 0
        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = self._ckpt_block(self.down_blocks[idx_b], hs[-1], temb, zemb)
                idx_b += 1
                attn = self.down_attns[idx_a]; idx_a += 1
                if attn is not None:
                    h = attn(h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = self._ckpt_block(self.down_resamples[idx_r], hs[-1], temb, zemb)
                idx_r += 1
                hs.append(h)

        # ---- bottleneck ----
        h = hs[-1]
        h = self._ckpt_block(self.mid_block1, h, temb, zemb)
        h = self.mid_attn(h)
        h = self._ckpt_block(self.mid_block2, h, temb, zemb)

        # ---- decoder ----
        idx_b = idx_a = idx_r = 0
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                cat = torch.cat([h, hs.pop()], dim=1)
                h = self._ckpt_block(self.up_blocks[idx_b], cat, temb, zemb)
                idx_b += 1
            attn = self.up_attns[idx_a]; idx_a += 1
            if attn is not None:
                h = attn(h)
            if i_level != 0:
                h = self._ckpt_block(self.up_resamples[idx_r], h, temb, zemb)
                idx_r += 1

        assert not hs

        h = self.act(self.final_gn(h))
        h = self.final_conv(h)
        if self.not_use_tanh:
            return h
        return torch.tanh(h)


# --------------------------------------------------------------------------- #
# 4. Discriminator3D_large — mirrors discriminator.Discriminator_large
# --------------------------------------------------------------------------- #
class TimestepEmbedding3d(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.main = nn.Sequential(
            dense_init(embedding_dim, hidden_dim),
            act,
            dense_init(hidden_dim, output_dim),
        )

    def forward(self, t):
        emb = get_timestep_embedding(t, self.embedding_dim)
        return self.main(emb)


class DownConvBlock3d(nn.Module):
    """3D analogue of discriminator.DownConvBlock. AvgPool3d for downsample
    (no FIR). Time-emb bias added between conv1 and conv2."""
    def __init__(self, in_channel, out_channel, *, kernel_size=3, padding=1,
                 t_emb_dim=128, downsample=False, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.downsample = downsample
        self.act = act
        self.conv1 = conv3x3x3(in_channel, out_channel, padding=padding)
        self.conv2 = conv3x3x3(out_channel, out_channel, padding=padding,
                               init_scale=0.0)
        self.dense_t1 = dense_init(t_emb_dim, out_channel)
        self.skip = conv1x1x1(in_channel, out_channel, bias=False)

    def forward(self, x, t_emb):
        out = self.act(x)
        out = self.conv1(out)
        out = out + self.dense_t1(t_emb)[..., None, None, None]
        out = self.act(out)
        if self.downsample:
            # AvgPool3d by 2 — same effect as 2D's downsample_2d FIR(1331)
            out = F.avg_pool3d(out, 2)
            x = F.avg_pool3d(x, 2)
        out = self.conv2(out)
        skip = self.skip(x)
        return (out + skip) / math.sqrt(2.0)


class Discriminator3D_large(nn.Module):
    """3D discriminator with time conditioning. 5 downsample stages
    (instead of 6 from the 2D version) because in 3D we go from 64^3
    rather than 256^2 — the input is already small. After 5 downs we land
    at 2^3, which is fine for the stddev pool + linear head."""

    def __init__(self, nc=2, ngf=32, t_emb_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.act = act
        self.t_embed = TimestepEmbedding3d(
            embedding_dim=t_emb_dim, hidden_dim=t_emb_dim,
            output_dim=t_emb_dim, act=act)

        self.start_conv = conv1x1x1(nc, ngf * 2)
        # 64 -> 32
        self.conv1 = DownConvBlock3d(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim,
                                     downsample=True, act=act)
        # 32 -> 16
        self.conv2 = DownConvBlock3d(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim,
                                     downsample=True, act=act)
        # 16 -> 8
        self.conv3 = DownConvBlock3d(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim,
                                     downsample=True, act=act)
        # 8 -> 4
        self.conv4 = DownConvBlock3d(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim,
                                     downsample=True, act=act)
        # 4 -> 2
        self.conv5 = DownConvBlock3d(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim,
                                     downsample=True, act=act)

        self.final_conv = conv3x3x3(ngf * 8 + 1, ngf * 8, padding=1)
        self.end_linear = dense_init(ngf * 8, 1)

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))
        h = torch.cat((x, x_t), dim=1)
        h = self.start_conv(h)
        h = self.conv1(h, t_embed)
        h = self.conv2(h, t_embed)
        h = self.conv3(h, t_embed)
        h = self.conv4(h, t_embed)
        out = self.conv5(h, t_embed)

        # stddev minibatch trick (same as 2D, with 3D dims)
        B, C, D, H, W = out.shape
        group = min(B, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat,
                          C // self.stddev_feat, D, H, W)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4, 5], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, D, H, W)
        out = torch.cat([out, stddev], dim=1)

        out = self.final_conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)
        return out
