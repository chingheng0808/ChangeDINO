import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torch import einsum


##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=dim)
        self.relu = nn.ReLU6()
        self.conv3 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        out = self.conv3(self.relu(self.conv2(self.bn2(self.conv1(self.bn1(x))))))
        return out


###############################################################
def to(x):
    return {"device": x.device, "dtype": x.dtype}


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head**-0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, "b (x y) c -> b x y c", x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h


class OCDA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        depth=None,
        window_size: int = 8,
        overlap_ratio: float = 0.5,
        bias=False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert depth is not None
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # OCA: overlapping unfold & relative position embedding ======
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        pad = (self.overlap_win_size - self.window_size) // 2
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=self.window_size,
            padding=pad,
        )

        # relative position embedding
        self.rel_pos_emb = RelPosEmb(
            block_size=self.window_size,
            rel_size=self.window_size + (self.overlap_win_size - self.window_size),
            dim_head=self.head_dim,
        )

        # qkv projection
        self.q_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        # temperature
        self.register_buffer("dot_scale", torch.tensor(self.head_dim**-0.5))

        # differential arch.
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )

        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(2 * self.head_dim))
        self.eps = 1e-5  # Epsilon for numerical stability
        nn.init.constant_(self.rms_scale, 1.0)

        self.out_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        # x: [B, C, H, W]
        B, _, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, f"H,W must be multiple of window_size={ws}"

        # ----- linear projections -----
        qs = self.q_proj(x)  # (B, 2C, H, W)
        ks = self.k_proj(x)  # (B, 2C, H, W)
        vs = self.v_proj(x)  # (B, 2C, H, W)

        # patchify to non-overlapping window (q)
        qs = rearrange(qs, "b c (hh p1) (ww p2) -> (b hh ww) (p1 p2) c", p1=ws, p2=ws)

        # patchify to overlapped window (k, v)
        J = self.overlap_win_size * self.overlap_win_size
        I = (H // ws) * (W // ws)
        ks = self.unfold(ks)  # (B, (2C)*J, I)
        vs = self.unfold(vs)  # (B, (2C)*J, I)
        ks = rearrange(ks, "b (c j) i -> (b i) j c", j=J)
        vs = rearrange(vs, "b (c j) i -> (b i) j c", j=J)

        # split heads
        qs, ks, vs = map(
            lambda t: rearrange(t, "b n (head c) -> (b head) n c", head=self.num_heads),
            (qs, ks, vs),
        )

        q1, q2 = qs.chunk(2, dim=-1)  # (B*I*head, ws^2, head_dim)
        k1, k2 = ks.chunk(2, dim=-1)  # (B*I*head, J,    head_dim)

        # logits
        logits1 = (
            q1 @ k1.transpose(-2, -1) * self.dot_scale
        )  # (B*I*head, ws^2, J), J=ws^2
        logits2 = q2 @ k2.transpose(-2, -1) * self.dot_scale  # (B*I*head, ws^2, J)

        # add relative positional embedding
        logits1 += self.rel_pos_emb(q1)
        logits2 += self.rel_pos_emb(q2)

        attn1 = logits1.softmax(dim=-1)
        attn2 = logits2.softmax(dim=-1)

        # lambda reparameter
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(
            q1
        )  # scalar
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(
            q2
        )  # scalar
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        lambda_full = lambda_full.repeat(B * I).unsqueeze(-1).unsqueeze(-1)

        # (attn1 - lambda*attn2) @ V
        out = (attn1 - lambda_full * attn2) @ vs  # (B*I*head, ws^2, 2*head_dim)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        out_rs = rearrange(
            out, "(b i head) n c -> (b head) (i n) c", b=B, i=I, head=self.num_heads
        )  # (B*head, H*W, 2*head_dim)

        # Compute RMSNorm
        rms_norm = torch.sqrt(
            out_rs.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # (B*num_heads, N, 1)
        out_norm = (out_rs / rms_norm) * self.rms_scale  # (B*num_heads, N, 2*d_head)

        # Reshape back to (B, head, N, 2*head_dim)
        out_norm = out_norm.view(B, self.num_heads, -1, 2 * self.head_dim)

        # Scale the normalized output
        out_norm = out_norm * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (B, 2C, H, W)
        out = out_norm.transpose(1, 2).contiguous().view(B, -1, H, W)

        # project out
        out = self.out_proj(out)  # (B, C, H, W)
        return out


class CDA(nn.Module):
    def __init__(self, dim, num_heads=4, depth=None, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        assert depth is not None
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        # temperature
        self.register_buffer("dot_scale", torch.tensor(self.head_dim**-0.5))

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(1, 1, 2 * self.head_dim))
        self.eps = 1e-5  # Epsilon for numerical stability
        nn.init.constant_(self.rms_scale, 1.0)

        self.out_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0, bias=bias)

    def _split_heads(self, t, head_dim=None):
        # (B, C, H, W) -> (B, h, N, d); N = H*W; C = h*d
        B = t.shape[0]
        if head_dim is None:
            head_dim = self.head_dim
        t = t.view(B, self.num_heads, head_dim, -1).permute(0, 1, 3, 2).contiguous()
        return t

    def forward(self, x):
        B, _, H, W = x.shape

        # linear projection
        Q = self.q_proj(x)  # (B, dim*2, H, W)
        K = self.k_proj(x)  # (B, dim*2, H, W)
        V = self.v_proj(x)  # (B, dim*2, H, W)

        Q1, Q2 = self._split_heads(Q, self.head_dim * 2).chunk(
            2, dim=-1
        )  # (B, h, N, d)
        K1, K2 = self._split_heads(K, self.head_dim * 2).chunk(
            2, dim=-1
        )  # (B, h, N, d)
        V = self._split_heads(V, self.head_dim * 2)  # (B, h, N, d*2)

        # logits
        logits1 = (Q1 @ K1.transpose(-2, -1)) * self.dot_scale  # (B, h, N, N)
        logits2 = (Q2 @ K2.transpose(-2, -1)) * self.dot_scale  # (B, h, N, N)

        attn1 = logits1.softmax(dim=-1)
        attn2 = logits2.softmax(dim=-1)

        # lambda
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(Q1)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(Q2)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        lambda_full = lambda_full.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        out = (attn1 - lambda_full * attn2) @ V  # (B, h, N, d*2)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        out_rs = out.contiguous().view(
            B * self.num_heads, H * W, 2 * self.head_dim
        )  # (batch*num_heads, N, 2*d_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(
            out_rs.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # (batch*num_heads, N, 1)
        out_norm = (
            out_rs / rms_norm
        ) * self.rms_scale  # (batch*num_heads, N, 2*d_head)

        # Reshape back to (batch, num_heads, N, 2 * d_head)
        out_norm = out_norm.view(B, self.num_heads, H * W, 2 * self.head_dim)

        # Scale the normalized output
        out_norm = out_norm * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (B, 2C, H, W)
        out = out_norm.transpose(2, 3).contiguous().view(B, -1, H, W)

        out = self.out_proj(out)  # (B, C, H, W)
        return out

## Multi-DConv Head Transposed Self-Attention
class ChA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        spatial_attn_type="OCDA",
        window_size=8,
        overlap_ratio=0.5,
        num_channel_heads=8,
        num_spatial_heads=4,
        depth=1,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super(TransformerBlock, self).__init__()

        if spatial_attn_type == "OCDA":
            self.spatial_attn = OCDA(
                dim, num_spatial_heads, depth, window_size, overlap_ratio, bias
            )
        elif spatial_attn_type == "CDA":
            self.spatial_attn = CDA(dim, num_spatial_heads, depth, bias)
        self.channel_attn = ChA(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor*dim)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor*dim)

    def forward(self, x):
        x = x + self.spatial_attn(self.norm1(x))
        x = x + self.spatial_ffn(self.norm2(x))
        x = x + self.channel_attn(self.norm3(x))
        x = x + self.channel_ffn(self.norm4(x))
        return x