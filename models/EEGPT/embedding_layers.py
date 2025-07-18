import torch
import torch.nn as nn
from typing import Tuple, Optional


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    This module converts an input image into a sequence of patches
    (along the width dimension).
    Each patch is flattened and projected into an embedding space.

    Attributes
    ----------
    img_size : Tuple[int]
        Size of the input image (height, width).
    patch_size : int
        Size of each patch.
    patch_stride : int
        Stride for patch extraction.
    num_patches : Tuple[int, int]
        Number of patches extracted from the input image.
        First element is the number of patches along the height,
        second element is the number of patches along the width.
    proj : nn.Conv2d
        Convolutional layer to extract patches and project them
        into the embedding space.
    """
    def __init__(
            self, img_size: Tuple[int],
            patch_size: int=16,
            patch_stride: Optional[int]=None,
            embed_dim: int=768
        ):
        """
        Parameters
        ----------
        img_size : tuple
            Size of the input image (height, width).
        patch_size : int, optional
            Size of each patch. \n
            Default is 16.
        patch_stride : int, optional
            Stride for patch extraction.
            If not given, it defaults to patch_size.
        embed_dim : int, optional
            Dimension of the embedding space. \n
            Default is 768.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_size if patch_stride is None else patch_stride

        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = (
                (img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        # temporal convolution to extract patches
        self.proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, self.patch_stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbed module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, T)
            where B is batch size,
            C is number of channels,
            and T is sequence length.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, num_patches, C, embed_dim)
            where num_patches is the number of patches
            extracted from the input image,
            which depends on the image size, patch size and patch stride.
        """
        x = x.unsqueeze(1)
        x = self.proj(x).transpose(1, 3)
        return x


class PatchNormEmbed(nn.Module):
    """
    Image-like data to Patch Embedding with Layer Normalisation.
    This module converts an input image-like tensor into
    a sequence of patches (along the width dimension).
    Each patch is flattened, normalised, and projected linearly
    into an embedding space.

    Attributes
    ----------
    img_size : Tuple[int]
        Size of the input image (height, width).
    patch_size : int
        Size of each patch.
    patch_stride : int
        Stride for patch extraction.
    num_patches : Tuple[int, int]
        Number of patches extracted from the input image.
        First element is the number of patches along the height,
        second element is the number of patches along the width.
    unfold : torch.nn.Unfold
        Unfold layer to extract patches from the input tensor.
    proj : nn.Linear
        Linear layer to project patches into the embedding space.
    """
    def __init__(
            self, img_size: Tuple[int],
            patch_size: int=16,
            patch_stride: Optional[int]=None,
            embed_dim: int=768,
            add_mean_std: bool = False
        ):
        """
        Parameters
        ----------
        img_size : tuple
            Size of the input image (height, width).
        patch_size : int, optional
            Size of each patch. \n
            Default is 16.
        patch_stride : int, optional
            Stride for patch extraction.
            If not given, it defaults to patch_size.
        embed_dim : int, optional
            Dimension of the embedding space. \n
            Default is 768.
        add_mean_std : bool, optional
            If True, adds mean and standard deviation
            of each patch to the output tensor.
            Default is False.
        """
        super().__init__()
        
        assert img_size[1] % patch_size==0
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_size if patch_stride is None else patch_stride
        self.add_mean_std = add_mean_std
        
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = (
                (img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.unfold = torch.nn.Unfold(
            kernel_size=(1, patch_size),
            stride = (1, self.patch_stride)
        )

        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, T)
            where B is batch size,
            C is number of channels,
            and T is sequence length.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, num_patches, C, embed_dim)
            where num_patches is the number of patches
            extracted from the input image,
            which depends on the image size, patch size and patch stride.
        """
        B, C, T = x.shape
        x = x.unsqueeze(1)   # (B, 1, C, T)

        x = self.unfold(x)   # (B, C * self.patch_size, num_patches)
        x = x.transpose(-1, -2)   # (B, num_patches, C * self.patch_size)
        
        x = x.view(B, C, -1, self.patch_size).contiguous()
        x = x.transpose(1, 2)

        if self.add_mean_std:
            # add mean and std to the patch embedding
            m = torch.mean(x, dim=-1).unsqueeze(-1)
            v = torch.std(x, dim=-1).unsqueeze(-1)

        x = torch.layer_norm(x, (self.patch_size,))

        if self.add_mean_std:
            x = torch.cat([x, m, v], dim=-1)

        x = self.proj(x)   # (B, T, C, D)

        return x

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.
    """
    def __init__(
            self,
            dim: int,
            theta: float=10000,
            learned_freq: bool=False,
            interpolate_factor: float=1.0
        ):
        """
        Rotary Positional Embedding module to encode
        sequential information into embeddings.
        
        Parameters
        ----------
        dim : int
            Dimension of the frequency embedding.
        theta : float, optional
            Scaling factor for the frequencies.
        learned_freq : bool, optional
            Whether the frequencies are learnable parameters.
            Default is False.
        interpolate_factor : float, optional
            Scaling factor for interpolated positional encoding.
            Default is 1.0, which means no interpolation.
        """
        super().__init__()
        assert interpolate_factor >= 1.0, "Interpolate factor must be >= 1.0"

        # Initialize frequency parameters
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)),
            requires_grad = learned_freq
        )

        self.interpolate_factor = interpolate_factor

        # Cache for prepared frequencies
        self.cache = {}

    def prepare_freqs(
            self,
            num_patches: Tuple[int, int],
            device: str='cuda',
            dtype: torch.dtype=torch.float32,
            offset: float=0.
        ) -> torch.Tensor:
        """
        Prepares the frequency embeddings for the given number of patches.

        Parameters
        ----------
        num_patches : Tuple[int, int]
            Tuple specifying the dimensions (C, N) where
            C is the channels and N is the number of positions.
        device : str, optional
            Device to store the frequencies on (e.g., 'cuda' or 'cpu').
            Default is 'cuda'.
        dtype : torch.dtype, optional
            Data type for the frequencies.
            Default is torch.float32.
        offset : float, optional
            Offset added to position indexes before scaling.
            Default is 0.0.

        Returns
        -------
        torch.Tensor
            Tensor containing the prepared frequency embeddings.
            Shape is (C * N, D) where D is the dimension of the frequencies.
        """
        C, N = num_patches
        cache_key = f'freqs:{num_patches}'
        
        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate sequence positions and apply offset and scale
        seq_pos = torch.arange(
            N, device=device, dtype=dtype).repeat_interleave(repeats=C)
        seq_pos = (seq_pos + offset) / self.interpolate_factor
        
        # Compute outer product of positions and frequencies,
        # then expand along the last dimension
        freqs_scaled = torch.outer(
            seq_pos.type(self.freqs.dtype), self.freqs
        ).repeat_interleave(repeats=2, dim=-1)
        
        # Cache and return the computed frequencies
        self.cache[cache_key] = freqs_scaled

        return freqs_scaled
