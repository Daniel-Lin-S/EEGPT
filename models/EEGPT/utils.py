import torch
from typing import List, Dict


def apply_mask(mask: torch.Tensor, x: torch.Tensor):
    """
    Apply a mask to a 4D tensor x,
    keeping only the patches specified by the mask.

    Parameters
    ----------
    mask : torch.Tensor
        A tensor of shape [mN, mC] containing
        indices of patches to keep.
        (The indices are in the range [0, N * C - 1],
        with the form n * C + c)
        mN - number of temporal windows to keep
        mC - number of channels to keep
        Alternatively, a tensor of shape [mN] containing
        indices of patches to keep.
    x : torch.Tensor
        A tensor of shape [B, N, C, D].
        B - batch size
        N - number of temporal windows
        C - number of channels
        D - feature dimension (dimension of the embedding)

    Returns
    -------
    masked_x : torch.Tensor
        A tensor of shape [B, mN, mC, D] containing only
        the patches specified by the mask.
    """    
    B, N, C, D = x.shape

    if len(mask.shape)==2:
        mN, mC = mask.shape
        mask_keep = mask.reshape((1, mN * mC, 1)).repeat((B, 1, D))
    elif len(mask.shape) == 1:
        mN = mask.shape[0]
        mask_keep = mask.reshape((1, mN, 1)).repeat((B, 1, D))
    else:
        raise ValueError(
            "Mask should be either 2D or 1D tensor."
        )

    masked_x = torch.gather(
        x.reshape((B, N * C, D)), dim=-2, index=mask_keep
    )

    if len(mask.shape) == 2:
        masked_x = masked_x.contiguous().view((B, mN, mC, D))
    elif len(mask.shape) == 1:
        masked_x = masked_x.contiguous().view((B, mN, D))

    return masked_x


def apply_mask_t(mask_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply a temporal mask to a 3D tensor x,
    keeping only the patches specified by the mask.

    Parameters
    ----------
    mask_t : torch.Tensor
        A tensor of shape [mP] containing indices of
        patches in [1, ..., N] to keep.
        mP - number of patches to keep
    x : torch.Tensor
        A tensor of shape [B, N, D].
        B - batch size
        N - number of patches
        D - feature dimension (dimension of the embedding)

    Returns
    -------
    masked_x : torch.Tensor
        A tensor of shape [B, mN, D] containing only
        the patches specified by the mask.
    """
    B, _, D = x.shape
    mN = mask_t.shape[0]
    
    mask_keep = mask_t.reshape((1,mN,1)).repeat((B, 1, D))
    masked_x = torch.gather(x, dim=1, index=mask_keep)

    return masked_x


def get_config(
        embed_dim: int,
        embed_num: int,
        depth: List[int],
        num_heads: int
    ) -> Dict[str, dict]:
    """
    Turn the model parameters into a dictionary
    for encoder, predictor and reconstructor.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embeddings. (positional and channel)
    embed_num : int
        Number of output tokens.
    depth : List[int]
        Depth of the encoder, predictor and reconstructor.
        Should be a list of three integers.
    num_heads : int
        Number of attention heads in the transformer
        blocks.
    
    Returns
    -------
    Dict[str, dict]
        A dictionary with keys 'encoder', 'predictor',
        and 'reconstructor', each containing a dictionary
        with the model parameters.
    """
    
    models_configs = {
            'encoder': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'depth': depth[0],
                    'num_heads': num_heads,
                },
            'predictor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'predictor_embed_dim': embed_dim,
                    'depth': depth[1],
                    'num_heads': num_heads,
                },
            'reconstructor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'reconstructor_embed_dim': embed_dim,
                    'depth': depth[2],
                    'num_heads': num_heads,
                },
    }

    return models_configs


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.

    return stats
