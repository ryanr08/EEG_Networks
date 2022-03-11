import numpy as np

import torch
import torch.nn.functional as F

from preprocessing import data_prep
from model_CGAN import generator, discriminator

torch.cuda.manual_seed(0)
torch.manual_seed(0)

temp_z_ = torch.randn(8460, 100)
temp_y_ = (torch.rand(8460, 1) * 4).type(torch.LongTensor).squeeze()
temp_y_ = F.one_hot(temp_y_.to(torch.int64), num_classes=4)
temp_z_, temp_y_ = temp_z_.cuda(),temp_y_.cuda()

# G_valinna = generator()
G_W = generator()
G_AC = generator()

# G_valinna_path = r'cDCGAN_results/cDCGAN_generator_param.pkl'
G_W_path = r'cWGAN_results/cWGAN_generator_param.pkl'
G_AC_path = r'cACGAN_results/cACGAN_generator_param.pkl'

# G_valinna.load_state_dict((torch.load(G_valinna_path)))
G_W.load_state_dict((torch.load(G_W_path)))
G_AC.load_state_dict((torch.load(G_AC_path)))

# G_valinna.cuda()
G_W.cuda()
G_AC.cuda()

# G_valinna.eval()
G_W.eval()
G_AC.eval()

# G_vanilla_images = G_valinna(temp_z_, temp_y_).detach().cpu().numpy()
G_W_images = G_W(temp_z_, temp_y_).detach().cpu().numpy()
G_AC_images = G_AC(temp_z_, temp_y_).detach().cpu().numpy()
temp_y_ = temp_y_.detach().cpu().numpy()

# np.save(r'fake_data/GAN',G_vanilla_images)
np.save(r'fake_data/WGAN',G_W_images)
np.save(r'fake_data/ACGAN',G_AC_images)
np.save(r'fake_data/label',temp_y_)