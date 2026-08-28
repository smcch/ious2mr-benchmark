"""
All 2D generator + discriminator architectures for US->MRI synthesis.
A. Pix2Pix 2D (Attention ResU-Net)
B. SwinPix2Pix 2D (Swin Transformer U-Net)
C. CycleGAN 2D (ResNet generator with InstanceNorm)
D. CUT 2D (U-Net with PatchNCE feature hooks)
E. Discriminators (conditional + unconditional multi-scale PatchGAN)
F. 3D Refiner (lightweight U-Net for volume refinement)

Author: Santiago Cepeda / BrainUS-AI
"""
import numpy as np
import tensorflow as tf
from common import SN, GN, _valid_groups, InstNorm2D, InstNorm3D

INIT = tf.random_normal_initializer(0.0, 0.02)
HE_INIT = tf.keras.initializers.HeNormal()


# #############################################################################
# A. Pix2Pix 2D (Attention ResU-Net)
# #############################################################################

def residual_block_2d(x, filters):
    """Two-conv residual block with LayerNorm."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, padding="same", kernel_initializer=INIT)(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    return tf.keras.layers.LeakyReLU(0.2)(tf.keras.layers.Add()([shortcut, x]))


def attention_gate_2d(skip, gating, filters):
    """Attention gate for skip connections."""
    theta = tf.keras.layers.Conv2D(
        filters, 1, padding="same", kernel_initializer=INIT, use_bias=False)(skip)
    phi = tf.keras.layers.Conv2D(
        filters, 1, padding="same", kernel_initializer=INIT, use_bias=False)(gating)
    psi = tf.keras.layers.Conv2D(
        1, 1, padding="same", kernel_initializer=INIT, use_bias=False)(
        tf.keras.layers.Activation("relu")(tf.keras.layers.Add()([theta, phi])))
    return tf.keras.layers.Multiply()(
        [skip, tf.keras.layers.Activation("sigmoid")(psi)])


def _enc_block_2d(x, f, norm=True):
    x = tf.keras.layers.Conv2D(
        f, 4, strides=2, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    if norm:
        x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    return residual_block_2d(x, f)


def _dec_block_2d(x, skip, f, dropout=False):
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(
        f, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    if dropout:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    att_skip = attention_gate_2d(skip, x, f // 2)
    x = tf.keras.layers.Concatenate()([x, att_skip])
    return residual_block_2d(x, f)


def build_pix2pix_generator_2d(input_channels=1, output_channels=1):
    """Attention ResU-Net generator -- 2D version."""
    inputs = tf.keras.Input((None, None, input_channels))

    e1 = _enc_block_2d(inputs, 64, norm=False)   # H/2
    e2 = _enc_block_2d(e1, 128)                    # H/4
    e3 = _enc_block_2d(e2, 256)                    # H/8
    e4 = _enc_block_2d(e3, 512)                    # H/16
    b = _enc_block_2d(e4, 512)                     # H/32
    b = residual_block_2d(b, 512)

    d4 = _dec_block_2d(b, e4, 512, dropout=True)
    d3 = _dec_block_2d(d4, e3, 256)
    d2 = _dec_block_2d(d3, e2, 128)
    d1 = _dec_block_2d(d2, e1, 64)

    out = tf.keras.layers.UpSampling2D(2)(d1)
    out = tf.keras.layers.Conv2D(
        output_channels, 3, padding="same", kernel_initializer=INIT,
        activation="tanh", dtype="float32")(out)
    return tf.keras.Model(inputs, out, name="Pix2Pix_G_2D")


# #############################################################################
# B. SwinPix2Pix 2D
# #############################################################################

class PatchEmbed2D(tf.keras.layers.Layer):
    """2D Patch Embedding: Conv2D with patch_size stride."""
    def __init__(self, patch_size=2, embed_dim=36, **kwargs):
        super().__init__(**kwargs)
        self.ps = patch_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.proj = tf.keras.layers.Conv2D(
            self.embed_dim, self.ps, strides=self.ps, padding="valid",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        super().build(input_shape)

    def call(self, x):
        return self.norm(self.proj(x))


class PatchMerging2D(tf.keras.layers.Layer):
    """2D Patch Merging: interleave 4 spatial positions, reduce with Dense."""
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.reduction = tf.keras.layers.Dense(2 * self.dim, use_bias=False)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        super().build(input_shape)

    def call(self, x):
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = self.norm(x)
        return self.reduction(x)


def window_partition_2d(x, window_size):
    """Partition 2D feature map into non-overlapping windows."""
    ws = window_size
    B = tf.shape(x)[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, [B, H // ws, ws, W // ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, ws * ws, C])


def window_reverse_2d(windows, window_size, H, W):
    """Reverse 2D window partition."""
    ws = window_size
    B = tf.shape(windows)[0] // ((H // ws) * (W // ws))
    x = tf.reshape(windows, [B, H // ws, W // ws, ws, ws, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [B, H, W, -1])


class WindowAttention2D(tf.keras.layers.Layer):
    """2D Window-based Multi-Head Self-Attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def build(self, input_shape):
        self.qkv = tf.keras.layers.Dense(self.dim * 3)
        self.proj = tf.keras.layers.Dense(self.dim)
        ws = self.ws
        num_pos = (2 * ws - 1) ** 2
        self.rel_pos_table = self.add_weight(
            name="rel_pos_table", shape=(num_pos, self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True)
        coords = np.stack(np.meshgrid(
            np.arange(ws), np.arange(ws), indexing='ij'), axis=-1)
        coords_flat = coords.reshape(-1, 2)
        rel = coords_flat[:, None, :] - coords_flat[None, :, :]
        rel += ws - 1
        rel[:, :, 0] *= (2 * ws - 1)
        self._rel_pos_index_np = rel.sum(-1).flatten().astype(np.int32)
        super().build(input_shape)

    def call(self, x, training=None):
        N = self.ws ** 2
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [-1, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = tf.cast(self.scale, q.dtype)
        attn = tf.matmul(q, k, transpose_b=True) * scale
        rel_idx = tf.constant(self._rel_pos_index_np)
        bias = tf.gather(self.rel_pos_table, rel_idx)
        bias = tf.reshape(bias, [N, N, self.num_heads])
        bias = tf.transpose(bias, [2, 0, 1])
        attn = attn + tf.cast(tf.expand_dims(bias, 0), attn.dtype)
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, N, self.dim])
        return self.proj(out)


class SwinTransformerBlock2D(tf.keras.layers.Layer):
    """2D Swin Transformer block with optional shifted windows."""
    def __init__(self, dim, num_heads, window_size, shift=False,
                 mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.ws = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

    def build(self, input_shape):
        d = self.dim
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention2D(d, self.ws, self.num_heads)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        mlp_hidden = int(d * self.mlp_ratio)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_hidden, activation="gelu"),
            tf.keras.layers.Dense(d)])
        super().build(input_shape)

    def call(self, x, training=None):
        H, W = x.shape[1], x.shape[2]
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            ss = self.shift_size
            x = tf.roll(x, shift=[-ss, -ss], axis=[1, 2])
        x = window_partition_2d(x, self.ws)
        x = self.attn(x, training=training)
        x = window_reverse_2d(x, self.ws, H, W)
        if self.shift_size > 0:
            ss = self.shift_size
            x = tf.roll(x, shift=[ss, ss], axis=[1, 2])
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinStage2D(tf.keras.layers.Layer):
    """2D Swin stage: N blocks + optional downsampling."""
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, downsample=True, **kwargs):
        super().__init__(**kwargs)
        self.swin_blocks = []
        for i in range(depth):
            self.swin_blocks.append(SwinTransformerBlock2D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift=(i % 2 == 1), mlp_ratio=mlp_ratio))
        self.downsample_layer = PatchMerging2D(dim) if downsample else None

    def call(self, x, training=None):
        for blk in self.swin_blocks:
            x = blk(x, training=training)
        x_out = x
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
        return x_out, x


class AttentionGate2DLayer(tf.keras.layers.Layer):
    """Attention gate for 2D decoder skip connections."""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.W_g = tf.keras.layers.Conv2D(
            self.channels, 1, use_bias=False, kernel_initializer=HE_INIT)
        self.W_x = tf.keras.layers.Conv2D(
            self.channels, 1, use_bias=False, kernel_initializer=HE_INIT)
        self.psi = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, 1, kernel_initializer=HE_INIT),
            tf.keras.layers.Activation("sigmoid"),
        ])
        super().build(input_shape)

    def call(self, skip, gating):
        g = self.W_g(gating)
        x = self.W_x(skip)
        attn = tf.nn.relu(g + x)
        attn = self.psi(attn)
        return skip * attn


class SwinDecoderBlock2D(tf.keras.layers.Layer):
    """Decoder block: UpSampling2D + Conv2D + AttentionGate + GroupNorm."""
    def __init__(self, out_ch, use_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = out_ch
        self.use_attention = use_attention

    def build(self, input_shape):
        self.up = tf.keras.layers.UpSampling2D(size=2)
        self.up_conv = tf.keras.layers.Conv2D(
            self.out_ch, 3, padding="same", kernel_initializer=HE_INIT,
            use_bias=False)
        self.up_gn = GN(32, channels=self.out_ch)
        if self.use_attention:
            self.attn_gate = AttentionGate2DLayer(self.out_ch)
        self.conv1 = tf.keras.layers.Conv2D(
            self.out_ch, 3, padding="same", kernel_initializer=HE_INIT,
            use_bias=False)
        self.gn1 = GN(32, channels=self.out_ch)
        self.conv2 = tf.keras.layers.Conv2D(
            self.out_ch, 3, padding="same", kernel_initializer=HE_INIT,
            use_bias=False)
        self.gn2 = GN(32, channels=self.out_ch)
        super().build(input_shape)

    def call(self, x, skip=None):
        x = tf.nn.relu(self.up_gn(self.up_conv(self.up(x))))
        if skip is not None:
            if x.shape[1:-1] != skip.shape[1:-1]:
                s = tf.shape(skip)
                x = x[:, :s[1], :s[2], :]
            if self.use_attention:
                skip = self.attn_gate(skip, x)
            x = tf.concat([x, skip], axis=-1)
        x = tf.nn.relu(self.gn1(self.conv1(x)))
        x = tf.nn.relu(self.gn2(self.conv2(x)))
        return x


class SwinGenerator2D(tf.keras.Model):
    """
    SwinUNETR-style generator for 2D:
      Input (128x128xC) -> PatchEmbed(2) -> 64x64x36
      Stage0: 64x64x36, 2 blocks -> skip0 -> Merge -> 32x32x72
      Stage1: 32x32x72, 2 blocks -> skip1 -> Merge -> 16x16x144
      Stage2: 16x16x144, 2 blocks -> bottleneck
      Decoder: 16->32 (+skip1) -> 32->64 (+skip0) -> 64->128 (+initial_skip)
      Output: 128x128xC_out
    """
    def __init__(self, input_channels=1, output_channels=1,
                 embed_dim=36, depths=(2, 2, 2), num_heads=(6, 6, 12),
                 window_size=4, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        C = embed_dim

        self.patch_embed = PatchEmbed2D(2, C)

        self.enc_stages = []
        dim = C
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            downsample = (i < len(depths) - 1)
            self.enc_stages.append(SwinStage2D(
                dim=dim, depth=d, num_heads=h, window_size=window_size,
                mlp_ratio=mlp_ratio, downsample=downsample))
            if downsample:
                dim *= 2

        self.bottleneck = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim, 3, padding="same",
                                    kernel_initializer=HE_INIT, use_bias=False),
            GN(32, channels=dim),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(dim, 3, padding="same",
                                    kernel_initializer=HE_INIT, use_bias=False),
            GN(32, channels=dim),
            tf.keras.layers.Activation("relu"),
        ])

        self.dec3 = SwinDecoderBlock2D(C * 2, use_attention=True)
        self.dec2 = SwinDecoderBlock2D(C, use_attention=True)
        self.dec1 = SwinDecoderBlock2D(C, use_attention=True)

        self.initial_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(C, 3, padding="same",
                                    kernel_initializer=HE_INIT, use_bias=False),
            GN(32, channels=C),
            tf.keras.layers.Activation("relu"),
        ])

        self.out_conv = tf.keras.layers.Conv2D(
            output_channels, 1, padding="same",
            kernel_initializer=HE_INIT, activation="tanh", dtype="float32")

    def call(self, x, training=None):
        skip0_init = self.initial_conv(x)
        x = self.patch_embed(x)

        skips = []
        for stage in self.enc_stages:
            feat, x = stage(x, training=training)
            skips.append(feat)

        x = self.bottleneck(skips[-1])
        x = self.dec3(x, skips[1])
        x = self.dec2(x, skips[0])
        x = self.dec1(x, skip0_init)
        return self.out_conv(x)


def build_swin_generator_2d(input_channels=1, output_channels=1):
    """Build 2D SwinPix2Pix generator."""
    return SwinGenerator2D(
        input_channels=input_channels, output_channels=output_channels,
        embed_dim=36, depths=(2, 2, 2), num_heads=(6, 6, 12),
        window_size=4, mlp_ratio=4.0)


# #############################################################################
# C. CycleGAN 2D (ResNet generator with InstanceNorm)
# #############################################################################

def _cyclegan_resblock_2d(x, filters):
    """Residual block with InstanceNorm for CycleGAN."""
    shortcut = x
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = InstNorm2D()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = InstNorm2D()(x)
    return tf.keras.layers.Add()([shortcut, x])


def build_cyclegan_generator_2d(input_channels=1, output_channels=1,
                                 base_ch=64, n_down=2, n_res=9):
    """ResNet generator with InstanceNorm -- standard CycleGAN."""
    inputs = tf.keras.Input((None, None, input_channels))

    # c7s1-64: initial convolution
    x = tf.keras.layers.Conv2D(
        base_ch, 7, padding="same", kernel_initializer=INIT, use_bias=False)(inputs)
    x = InstNorm2D()(x)
    x = tf.keras.layers.ReLU()(x)

    # Downsampling
    ch = base_ch
    for _ in range(n_down):
        ch *= 2
        x = tf.keras.layers.Conv2D(
            ch, 3, strides=2, padding="same",
            kernel_initializer=INIT, use_bias=False)(x)
        x = InstNorm2D()(x)
        x = tf.keras.layers.ReLU()(x)

    # ResBlocks at bottleneck
    for _ in range(n_res):
        x = _cyclegan_resblock_2d(x, ch)

    # Upsampling
    for _ in range(n_down):
        ch //= 2
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2D(
            ch, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
        x = InstNorm2D()(x)
        x = tf.keras.layers.ReLU()(x)

    # c7s1-C_out: final convolution
    out = tf.keras.layers.Conv2D(
        output_channels, 7, padding="same", kernel_initializer=INIT,
        activation="tanh", dtype="float32")(x)
    return tf.keras.Model(inputs, out, name="CycleGAN_G_2D")


# #############################################################################
# D. CUT 2D (U-Net with PatchNCE feature hooks)
# #############################################################################

def build_cut_generator_2d(input_channels=1, output_channels=1):
    """U-Net generator returning [output, feat0, feat1, feat2, feat3] for PatchNCE.
    feat0: after initial conv (128x128)
    feat1: after e1 (64x64)
    feat2: after e2 (32x32)
    feat3: bottleneck (4x4)
    """
    inputs = tf.keras.Input((None, None, input_channels))

    # feat0: initial conv at full resolution
    feat0 = tf.keras.layers.Conv2D(
        64, 7, padding="same", kernel_initializer=INIT, use_bias=False)(inputs)
    feat0 = tf.keras.layers.LayerNormalization()(feat0)
    feat0 = tf.keras.layers.LeakyReLU(0.2)(feat0)

    # Encoder with feature extraction
    e1 = _enc_block_2d(feat0, 64, norm=False)     # H/2 = 64
    feat1 = e1
    e2 = _enc_block_2d(e1, 128)                    # H/4 = 32
    feat2 = e2
    e3 = _enc_block_2d(e2, 256)                    # H/8 = 16
    e4 = _enc_block_2d(e3, 512)                    # H/16 = 8

    # Bottleneck
    b = _enc_block_2d(e4, 512)                     # H/32 = 4
    b = residual_block_2d(b, 512)
    feat3 = b

    # Decoder with attention gates
    d4 = _dec_block_2d(b, e4, 512, dropout=True)
    d3 = _dec_block_2d(d4, e3, 256)
    d2 = _dec_block_2d(d3, e2, 128)
    d1 = _dec_block_2d(d2, e1, 64)

    out = tf.keras.layers.UpSampling2D(2)(d1)
    out = tf.keras.layers.Conv2D(
        output_channels, 3, padding="same", kernel_initializer=INIT,
        activation="tanh", dtype="float32")(out)

    return tf.keras.Model(inputs, [out, feat0, feat1, feat2, feat3],
                          name="CUT_G_2D")


# #############################################################################
# E. Discriminators
# #############################################################################

def build_conditional_disc_2d(input_channels=1, target_channels=1, name="disc"):
    """Conditional PatchGAN 2D (for Pix2Pix/SwinPix2Pix).
    Takes [input, target] -> concat -> layers.
    Returns [output] + features for FM loss.
    """
    inp = tf.keras.Input(shape=(None, None, input_channels), name="input_image")
    tar = tf.keras.Input(shape=(None, None, target_channels), name="target_image")
    x = tf.keras.layers.Concatenate()([inp, tar])
    features = []

    # Layer 1: no norm
    x = SN(tf.keras.layers.Conv2D(
        64, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    # Layer 2
    x = SN(tf.keras.layers.Conv2D(
        128, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    # Layer 3
    x = SN(tf.keras.layers.Conv2D(
        256, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    # Layer 4: stride=1
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = SN(tf.keras.layers.Conv2D(
        512, 4, strides=1, padding="valid",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    # Output: stride=1
    x = tf.keras.layers.ZeroPadding2D()(x)
    out = SN(tf.keras.layers.Conv2D(
        1, 4, strides=1, padding="valid",
        kernel_initializer=INIT, dtype="float32"))(x)

    return tf.keras.Model(inputs=[inp, tar], outputs=[out] + features, name=name)


def build_unconditional_disc_2d(input_channels=1, name="disc"):
    """Unconditional PatchGAN 2D (for CycleGAN/CUT).
    Takes single image -> layers.
    Returns [output] + features for FM loss.
    """
    inp = tf.keras.Input(shape=(None, None, input_channels), name="input_image")
    x = inp
    features = []

    # Layer 1: no norm
    x = SN(tf.keras.layers.Conv2D(
        64, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    # Layer 2
    x = SN(tf.keras.layers.Conv2D(
        128, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    # Layer 3
    x = SN(tf.keras.layers.Conv2D(
        256, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    # Output: stride=1
    x = tf.keras.layers.ZeroPadding2D()(x)
    out = SN(tf.keras.layers.Conv2D(
        1, 4, strides=1, padding="valid",
        kernel_initializer=INIT, dtype="float32"))(x)

    return tf.keras.Model(inputs=inp, outputs=[out] + features, name=name)


class MultiScaleCondDisc2D(tf.keras.Model):
    """2-scale conditional discriminator: original + 2x downsampled."""
    def __init__(self, input_channels=1, target_channels=1, num_scales=2, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = num_scales
        self.disc_list = []
        self.pool_list = []
        for s in range(num_scales):
            self.disc_list.append(
                build_conditional_disc_2d(input_channels, target_channels,
                                          f"disc_scale_{s}"))
            if s > 0:
                self.pool_list.append(
                    tf.keras.layers.AveragePooling2D(2 ** s, padding="same"))

    def call(self, inputs, training=None):
        inp, tar = inputs
        outputs = []
        for s in range(self.num_scales):
            if s == 0:
                inp_s, tar_s = inp, tar
            else:
                inp_s = self.pool_list[s - 1](inp)
                tar_s = self.pool_list[s - 1](tar)
            outputs.append(self.disc_list[s]([inp_s, tar_s], training=training))
        return outputs


class MultiScaleUncondDisc2D(tf.keras.Model):
    """2-scale unconditional discriminator: original + 2x downsampled."""
    def __init__(self, input_channels=1, num_scales=2, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = num_scales
        self.disc_list = []
        self.pool_list = []
        for s in range(num_scales):
            self.disc_list.append(
                build_unconditional_disc_2d(input_channels, f"disc_scale_{s}"))
            if s > 0:
                self.pool_list.append(
                    tf.keras.layers.AveragePooling2D(2 ** s, padding="same"))

    def call(self, inputs, training=None):
        outputs = []
        for s in range(self.num_scales):
            if s == 0:
                inp_s = inputs
            else:
                inp_s = self.pool_list[s - 1](inputs)
            outputs.append(self.disc_list[s](inp_s, training=training))
        return outputs


# #############################################################################
# F. 3D Refiner (lightweight U-Net for volume refinement)
# #############################################################################

def _residual_block_3d(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv3D(
            filters, 1, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = tf.keras.layers.Conv3D(
        filters, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv3D(
        filters, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    return tf.keras.layers.LeakyReLU(0.2)(tf.keras.layers.Add()([shortcut, x]))


def build_3d_refiner(input_channels=1):
    """Lightweight 3D U-Net for volume refinement. Residual learning."""
    inputs = tf.keras.Input((None, None, None, input_channels))

    # Encoder: 2 downsamples
    e1 = tf.keras.layers.Conv3D(
        16, 3, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False)(inputs)
    e1 = tf.keras.layers.LayerNormalization()(e1)
    e1 = tf.keras.layers.LeakyReLU(0.2)(e1)

    e2 = tf.keras.layers.Conv3D(
        32, 3, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False)(e1)
    e2 = tf.keras.layers.LayerNormalization()(e2)
    e2 = tf.keras.layers.LeakyReLU(0.2)(e2)

    # Bottleneck
    b = _residual_block_3d(e2, 32)
    b = _residual_block_3d(b, 32)

    # Decoder
    d1 = tf.keras.layers.UpSampling3D(size=2)(b)
    d1 = tf.keras.layers.Conv3D(
        16, 3, padding="same", kernel_initializer=INIT, use_bias=False)(d1)
    d1 = tf.keras.layers.LayerNormalization()(d1)
    d1 = tf.keras.layers.ReLU()(d1)
    d1 = tf.keras.layers.Concatenate()([d1, e1])
    d1 = tf.keras.layers.Conv3D(
        16, 3, padding="same", kernel_initializer=INIT, use_bias=False)(d1)
    d1 = tf.keras.layers.LayerNormalization()(d1)
    d1 = tf.keras.layers.ReLU()(d1)

    d0 = tf.keras.layers.UpSampling3D(size=2)(d1)
    d0 = tf.keras.layers.Conv3D(
        16, 3, padding="same", kernel_initializer=INIT, use_bias=False)(d0)
    d0 = tf.keras.layers.LayerNormalization()(d0)
    d0 = tf.keras.layers.ReLU()(d0)

    # Residual output
    residual = tf.keras.layers.Conv3D(
        input_channels, 3, padding="same",
        kernel_initializer=INIT, dtype="float32")(d0)
    output = tf.keras.layers.Add()([inputs, residual])
    output = tf.keras.layers.Lambda(
        lambda x: tf.clip_by_value(x, -1.0, 1.0))(output)

    return tf.keras.Model(inputs=inputs, outputs=output, name="Refiner3D")


# #############################################################################
# G. 3D Generators (for full 3D experiment variants)
# #############################################################################

def _enc_block_3d(x, f, norm=True):
    x = tf.keras.layers.Conv3D(
        f, 4, strides=2, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    if norm:
        x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    return _residual_block_3d(x, f)


def _dec_block_3d(x, skip, f, dropout=False):
    x = tf.keras.layers.UpSampling3D(2)(x)
    x = tf.keras.layers.Conv3D(
        f, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    if dropout:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    # Attention gate
    theta = tf.keras.layers.Conv2D if False else tf.keras.layers.Conv3D
    theta_x = tf.keras.layers.Conv3D(
        f // 2, 1, padding="same", kernel_initializer=INIT, use_bias=False)(skip)
    phi_g = tf.keras.layers.Conv3D(
        f // 2, 1, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    psi = tf.keras.layers.Conv3D(
        1, 1, padding="same", kernel_initializer=INIT, use_bias=False)(
        tf.keras.layers.Activation("relu")(tf.keras.layers.Add()([theta_x, phi_g])))
    att_skip = tf.keras.layers.Multiply()(
        [skip, tf.keras.layers.Activation("sigmoid")(psi)])
    x = tf.keras.layers.Concatenate()([x, att_skip])
    return _residual_block_3d(x, f)


def build_pix2pix_generator_3d(input_channels=1, output_channels=1,
                                 patch_size=None):
    """3D Attention ResU-Net generator for patch-based training.
    patch_size=None uses fully convolutional (None,None,None,ch) input.
    """
    if patch_size is None:
        inputs = tf.keras.Input((None, None, None, input_channels))
    else:
        inputs = tf.keras.Input((patch_size, patch_size, patch_size, input_channels))

    e1 = _enc_block_3d(inputs, 32, norm=False)  # 32
    e2 = _enc_block_3d(e1, 64)                   # 16
    e3 = _enc_block_3d(e2, 128)                  # 8
    b = _enc_block_3d(e3, 256)                   # 4
    b = _residual_block_3d(b, 256)

    d3 = _dec_block_3d(b, e3, 128, dropout=True)
    d2 = _dec_block_3d(d3, e2, 64)
    d1 = _dec_block_3d(d2, e1, 32)

    out = tf.keras.layers.UpSampling3D(2)(d1)
    out = tf.keras.layers.Conv3D(
        output_channels, 3, padding="same", kernel_initializer=INIT,
        activation="tanh", dtype="float32")(out)
    return tf.keras.Model(inputs, out, name="Pix2Pix_G_3D")


def _cyclegan_resblock_3d(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv3D(
        filters, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = InstNorm3D()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(
        filters, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
    x = InstNorm3D()(x)
    return tf.keras.layers.Add()([shortcut, x])


def build_cyclegan_generator_3d(input_channels=1, output_channels=1,
                                 base_ch=32, n_down=3, n_res=6, patch_size=None):
    """3D CycleGAN ResNet generator with InstanceNorm and skip connections.
    patch_size=None uses fully convolutional (None,None,None,ch) input.
    """
    if patch_size is None:
        inputs = tf.keras.Input((None, None, None, input_channels))
    else:
        inputs = tf.keras.Input((patch_size, patch_size, patch_size, input_channels))

    # Initial conv
    x = tf.keras.layers.Conv3D(
        base_ch, 7, padding="same", kernel_initializer=INIT, use_bias=False)(inputs)
    x = InstNorm3D()(x)
    x = tf.keras.layers.ReLU()(x)
    skips = [x]

    # Downsampling
    ch = base_ch
    for _ in range(n_down):
        ch *= 2
        x = tf.keras.layers.Conv3D(
            ch, 3, strides=2, padding="same",
            kernel_initializer=INIT, use_bias=False)(x)
        x = InstNorm3D()(x)
        x = tf.keras.layers.ReLU()(x)
        skips.append(x)

    # ResBlocks
    for _ in range(n_res):
        x = _cyclegan_resblock_3d(x, ch)

    # Upsampling with skip connections
    for i in range(n_down):
        ch //= 2
        x = tf.keras.layers.UpSampling3D(2)(x)
        x = tf.keras.layers.Conv3D(
            ch, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
        x = InstNorm3D()(x)
        x = tf.keras.layers.ReLU()(x)
        # Skip connection
        skip = skips[-(i + 2)]
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv3D(
            ch, 1, padding="same", kernel_initializer=INIT, use_bias=False)(x)
        x = InstNorm3D()(x)
        x = tf.keras.layers.ReLU()(x)

    out = tf.keras.layers.Conv3D(
        output_channels, 7, padding="same", kernel_initializer=INIT,
        activation="tanh", dtype="float32")(x)
    return tf.keras.Model(inputs, out, name="CycleGAN_G_3D")


def build_cut_generator_3d(input_channels=1, output_channels=1,
                            base_ch=32, n_down=3, n_res=6, patch_size=None):
    """3D U-Net generator for CUT with feature hooks.
    patch_size=None uses fully convolutional (None,None,None,ch) input.
    """
    if patch_size is None:
        inputs = tf.keras.Input((None, None, None, input_channels))
    else:
        inputs = tf.keras.Input((patch_size, patch_size, patch_size, input_channels))

    # feat0: initial conv
    feat0 = tf.keras.layers.Conv3D(
        base_ch, 7, padding="same", kernel_initializer=INIT, use_bias=False)(inputs)
    feat0 = InstNorm3D()(feat0)
    feat0 = tf.keras.layers.ReLU()(feat0)

    # Encoder
    x = feat0
    ch = base_ch
    enc_skips = [feat0]
    enc_feats = [feat0]

    for i in range(n_down):
        ch *= 2
        x = tf.keras.layers.Conv3D(
            ch, 3, strides=2, padding="same",
            kernel_initializer=INIT, use_bias=False)(x)
        x = InstNorm3D()(x)
        x = tf.keras.layers.ReLU()(x)
        enc_skips.append(x)
        if i < 2:
            enc_feats.append(x)

    # ResBlocks at bottleneck
    for _ in range(n_res):
        x = _cyclegan_resblock_3d(x, ch)
    enc_feats.append(x)  # bottleneck feature

    # Decoder
    for i in range(n_down):
        ch //= 2
        x = tf.keras.layers.UpSampling3D(2)(x)
        x = tf.keras.layers.Conv3D(
            ch, 3, padding="same", kernel_initializer=INIT, use_bias=False)(x)
        x = InstNorm3D()(x)
        x = tf.keras.layers.ReLU()(x)
        skip = enc_skips[-(i + 2)]
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv3D(
            ch, 1, padding="same", kernel_initializer=INIT, use_bias=False)(x)
        x = InstNorm3D()(x)
        x = tf.keras.layers.ReLU()(x)

    out = tf.keras.layers.Conv3D(
        output_channels, 7, padding="same", kernel_initializer=INIT,
        activation="tanh", dtype="float32")(x)

    # Return output + 4 encoder features for PatchNCE
    return tf.keras.Model(inputs, [out] + enc_feats, name="CUT_G_3D")


# #############################################################################
# H. 3D Discriminators
# #############################################################################

def build_conditional_disc_3d(patch_size=None, input_channels=1, target_channels=1,
                               name="disc"):
    """Conditional PatchGAN 3D. patch_size=None for fully convolutional."""
    ps = patch_size
    if ps is None:
        inp = tf.keras.Input((None, None, None, input_channels), name="input")
        tar = tf.keras.Input((None, None, None, target_channels), name="target")
    else:
        inp = tf.keras.Input((ps, ps, ps, input_channels), name="input")
        tar = tf.keras.Input((ps, ps, ps, target_channels), name="target")
    x = tf.keras.layers.Concatenate()([inp, tar])
    features = []

    x = SN(tf.keras.layers.Conv3D(
        64, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    x = SN(tf.keras.layers.Conv3D(
        128, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    x = SN(tf.keras.layers.Conv3D(
        256, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    x = tf.keras.layers.ZeroPadding3D()(x)
    x = SN(tf.keras.layers.Conv3D(
        512, 4, strides=1, kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    x = tf.keras.layers.ZeroPadding3D()(x)
    out = SN(tf.keras.layers.Conv3D(
        1, 4, strides=1, kernel_initializer=INIT, dtype="float32"))(x)

    return tf.keras.Model([inp, tar], [out] + features, name=name)


def build_unconditional_disc_3d(patch_size=None, input_channels=1, name="disc"):
    """Unconditional PatchGAN 3D. patch_size=None for fully convolutional."""
    ps = patch_size
    if ps is None:
        inp = tf.keras.Input((None, None, None, input_channels), name="input")
    else:
        inp = tf.keras.Input((ps, ps, ps, input_channels), name="input")
    x = inp
    features = []

    x = SN(tf.keras.layers.Conv3D(
        64, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    x = SN(tf.keras.layers.Conv3D(
        128, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    x = SN(tf.keras.layers.Conv3D(
        256, 4, strides=2, padding="same",
        kernel_initializer=INIT, use_bias=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    features.append(x)

    x = tf.keras.layers.ZeroPadding3D()(x)
    out = SN(tf.keras.layers.Conv3D(
        1, 4, strides=1, kernel_initializer=INIT, dtype="float32"))(x)

    return tf.keras.Model(inp, [out] + features, name=name)


class MultiScaleCondDisc3D(tf.keras.Model):
    """Multi-scale conditional 3D discriminator."""
    def __init__(self, patch_size=None, input_channels=1, target_channels=1,
                 num_scales=1, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = num_scales
        self.disc_list = []
        self.pool_list = []
        for s in range(num_scales):
            if patch_size is None:
                ps = None
            else:
                ps = patch_size // (2 ** s)
            self.disc_list.append(build_conditional_disc_3d(
                ps, input_channels, target_channels, f"disc_{s}"))
            if s > 0:
                self.pool_list.append(
                    tf.keras.layers.AveragePooling3D(2 ** s, padding="same"))

    def call(self, inputs, training=None):
        inp, tar = inputs
        outs = []
        for s in range(self.num_scales):
            if s == 0:
                i, t = inp, tar
            else:
                i = self.pool_list[s - 1](inp)
                t = self.pool_list[s - 1](tar)
            outs.append(self.disc_list[s]([i, t], training=training))
        return outs


class MultiScaleUncondDisc3D(tf.keras.Model):
    """Multi-scale unconditional 3D discriminator."""
    def __init__(self, patch_size=None, input_channels=1, num_scales=1, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = num_scales
        self.disc_list = []
        self.pool_list = []
        for s in range(num_scales):
            if patch_size is None:
                ps = None
            else:
                ps = patch_size // (2 ** s)
            self.disc_list.append(build_unconditional_disc_3d(
                ps, input_channels, f"disc_{s}"))
            if s > 0:
                self.pool_list.append(
                    tf.keras.layers.AveragePooling3D(2 ** s, padding="same"))

    def call(self, inputs, training=None):
        outs = []
        for s in range(self.num_scales):
            if s == 0:
                i = inputs
            else:
                i = self.pool_list[s - 1](inputs)
            outs.append(self.disc_list[s](i, training=training))
        return outs
