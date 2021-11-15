import torch
from torchvision.utils import save_image

folder = 'pre_fixed_set'

a = torch.load(f'{folder}/fixed_source.pkl')
save_image((a+1) / 2, f'{folder}/fixed_source.png', pad_value=255, normalize=False)

a = torch.load(f'{folder}/fixed_target.pkl')
save_image((a+1) / 2, f'{folder}/fixed_target.png', pad_value=255, normalize=False)

a = torch.load(f'{folder}/t_fixed_source.pkl')
save_image((a+1) / 2, f'{folder}/t_fixed_source.png', pad_value=255, normalize=False)

a = torch.load(f'{folder}/t_fixed_target.pkl')
save_image((a+1) / 2, f'{folder}/t_fixed_target.png', pad_value=255, normalize=False)