from models.EEGPT import EEGPTPretrain, MODELS_CONFIGS, get_config
import torch

model = EEGPTPretrain(get_config(**MODELS_CONFIGS['tiny1']))
B, C, T = 2, 58, 256*4
x = torch.randn(B, C, T)

mask_x, mask_y = model.make_masks(model.encoder.num_patches)
print('Shape of mask_x:', mask_x.shape)
print('Shape of mask_y:', mask_y.shape)

h, y = model.forward_target(x, mask_y)
print('Shape of h:', h.shape)
print('Shape of y:', y.shape)

z, r = model.forward_context(x, mask_x, mask_y)
print('Shape of z:', z.shape)
print('Shape of r:', r.shape)
