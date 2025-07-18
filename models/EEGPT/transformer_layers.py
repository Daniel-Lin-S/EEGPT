import torch
import torch.nn as nn
import math
from typing import Optional


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the last dimension of the input tensor `x` by 90 degrees.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be rotated.
        Should have shape (..., feature_dim).
        feature_dim - the number of features to rotate.
    
    Returns
    -------
    torch.Tensor
        The rotated tensor with the same shape as `x`.
        The last dimension is reshaped to have two halves,
        and the second half is negated.
    """
    x = x.reshape((*x.shape[:-1], x.shape[-1] // 2, 2))
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)

    return x.flatten(-2)


def apply_rotary_emb(
        freqs: torch.Tensor,
        t: torch.Tensor,
        start_index: int = 0, scale: float = 1.
    ):
    """
    Applies Rotary Position Embedding (RoPE)
    to the input tensor `t` using the provided frequencies `freqs`.

    Parameters
    ----------
    freqs : torch.Tensor
        The frequency tensor used for the RoPE.
        Should have shape (..., rot_dim).
        rot_dim - the number of dimensions to rotate.
    t : torch.Tensor
        The input tensor to which RoPE is applied.
        Should have shape (..., feature_dim).
    start_index : int, optional
        The starting index in the feature dimension where RoPE is applied.
        Default is 0.
    scale : float, optional
        A scaling factor applied to the RoPE.
        Default is 1.0.
    """
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    if rot_dim > t.shape[-1]:
        raise ValueError(
            f'rot_dim {rot_dim} is larger than feature dimension {t.shape[-1]}'
            '. Not enough features to apply RoPE.'
        )
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) module.
    It consists of two linear layers with an activation function
    and dropout in between.
    """
    def __init__(
            self, in_features: int,
            hidden_features: Optional[int]=None,
            out_features: Optional[int]=None,
            act_layer: nn.Module=nn.GELU,
            drop: float=0.
        ):
        """
        Parameters
        ----------
        in_features : int
            The number of input features.
        hidden_features : int, optional
            The number of hidden features.
            If not provided, it defaults to `in_features`.
        out_features : int, optional
            The number of output features.
            If not provided, it defaults to `in_features`.
        act_layer : nn.Module, optional
            The activation function to use between the layers.
            Default is nn.GELU.
        drop : float, optional
            Dropout rate applied after the first linear layer.
            Default is 0.0 (no dropout).
        """
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (..., in_features)
            where ... can be any number of dimensions.
    
        Returns
        -------
        torch.Tensor
            A tensor of shape (..., out_features)
            after applying the MLP.
            The shape is the same as the input tensor `x`
            except for the last dimension.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Each sample in the batch is either dropped (set to 0) or kept 
    (but scaled to maintain energy)
    """
    def __init__(self, drop_prob=None):
        """
        Parameters
        ----------
        drop_prob : float, optional
            Probability of dropping a path. Default is 0.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def drop_path(
            self, x: torch.Tensor,
            drop_prob: float = 0., training: bool = False
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, ...)
            where B is batch size and ... can be any number of dimensions.
        drop_prob : float, optional
            Probability of dropping a path. Default is 0.
        training : bool, optional
            Whether the model is in training mode. Default is False.
            Paths are only dropped during training.
        
        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input x,
            with paths dropped according to drop_prob.
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, ...)
            any number of dimensions after the batch dimension is allowed.
        
        Return
        -------
        torch.Tensor
            Output tensor with shape (B, C, T) after applying DropPath.
        """
        return self.drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    """
    Multi-Head Self Attention Module with optional RoPE and causal masking.
    """
    def __init__(
            self, dim: int,
            num_heads: int,
            qkv_bias: bool=False,
            attn_drop: float=0.,
            proj_drop: float=0.,
            is_causal: bool=False,
            use_rope: bool=False,
            return_attention: bool=False
        ):
        """
        Parameters
        ----------
        dim : int
            The input feature dimension.
        num_heads : int
            The number of attention heads.
        qkv_bias : bool, optional
            Whether to add a bias term to the query, key, and value projections.
            Default is False.
        attn_drop : float, optional
            Dropout rate for the attention weights.
            Default is 0.0 (no dropout).
        proj_drop : float, optional
            Dropout rate for the output projection.
            Default is 0.0 (no dropout).
        is_causal : bool, optional
            Whether to apply causal masking to the attention weights.
            Default is False.
        use_rope : bool, optional
            Whether to use Rotary Position Embedding (RoPE).
            Default is False.
        return_attention : bool, optional
            Whether to return the attention weights instead of the output.
            Default is False.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.return_attention= return_attention

    def forward(
            self, x: torch.Tensor,
            freqs: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C), where B is batch size,
            T is sequence length, and C is the feature dimension.
        freqs : torch.Tensor, optional
            Frequencies for RoPE, if applicable.
            Should have shape (T, C // num_heads).
            Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of with the same shape as x
            after applying multi-head self-attention.
        """
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(
            B, T, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)    # (3, B, num_heads, T, D)

        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, num_heads, T, D)
        
        if self.use_rope:
            if freqs is None:
                raise ValueError(
                    'freqs must be provided when use_rope is True.'
                )
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(
                    q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
                attn_maak = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_maak.masked_fill(
                    torch.logical_not(attn_mask), -float('inf'))
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask,
                    dim=-1
                )
            else:
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))),
                    dim=-1
                )

            return attn_weight

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0,
            is_causal=self.is_causal
        )    # (B, num_heads, T, hs)
        x = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, hs * num_heads)
        x = self.proj(x)   # (B, T, C)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block with Multi-Head Self Attention and MLP.

    Architecture:
    Multi-Head attention -> LayerNorm -> DropPath
    -> MLP -> LayerNorm -> DropPath \n 
    where DropPath is a regularisation technique that
    drops the entire sample (path) by chance. 
    """
    def __init__(
            self,
            dim: int,
            num_heads: int=8,
            mlp_ratio: float=4.,
            qkv_bias: bool=False,
            drop: float=0.,
            attn_drop: float=0.,
            drop_path: float=0.,
            act_layer: nn.Module=nn.GELU,
            norm_layer: nn.Module=nn.LayerNorm,
            is_causal: bool=False,
            use_rope: bool=False,
            return_attention: bool=False
        ):
        """
        Parameters
        ----------
        dim : int
            The input feature dimension.
        num_heads : int, optional
            The number of attention heads. Default is 8.
        mlp_ratio : float, optional
            The ratio of the hidden dimension in the MLP to the input dimension.
            Default is 4.0.
        qkv_bias : bool, optional
            Whether to add a bias term to the query, key, and value projections.
            Default is False.
        drop : float, optional
            Dropout rate for the output projection.
            Default is 0.0 (no dropout).
        attn_drop : float, optional
            Dropout rate for the attention weights.
            Default is 0.0 (no dropout).
        drop_path : float, optional
            Drop probability for the DropPath regularisation.
            Default is 0.0 (no DropPath).
        act_layer : nn.Module, optional
            Activation function used in the MLP.
            Default is nn.GELU.
        norm_layer : nn.Module, optional
            Normalisation layer applied before the attention and MLP.
            Default is nn.LayerNorm.
        is_causal : bool, optional
            Whether to apply causal masking to the attention weights.
            Default is False.
        use_rope : bool, optional
            Whether to use Rotary Position Embedding (RoPE)
            for the attention mechanism.
            Default is False.
        return_attention : bool, optional
            Whether to return the attention weights
            instead of the output.
            Default is False.
        """
        super().__init__()
        
        self.return_attention= return_attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            is_causal=is_causal, use_rope=use_rope,
            return_attention=return_attention
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(
            self, x: torch.Tensor,
            freqs: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C),
            B - batch size,
            T - sequence length,
            C - the feature dimension.
        freqs : torch.Tensor, optional
            Frequencies for RoPE, if applicable.
            Should have shape (T, C // num_heads).
            Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as x
            after applying the transformer block.
            If `return_attention` is True,
            returns the attention weights of shape
            (B, num_heads, T, T) instead of the output tensor.
        """
        y = self.attn(self.norm1(x), freqs)

        if self.return_attention: return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
