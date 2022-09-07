import torch
from torch.nn.utils import parameters_to_vector

from models import VAE, SimpleHalMoonNN
from dataloader import load_data

model_dir = './pretrained/'
use_gpu = True

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

train_dataloader, test_dataloader = load_data('HalfMoon', batch_size=256)

# Load saved model
clf = SimpleHalMoonNN()
clf.load_state_dict(torch.load(model_dir + 'model_reference.pth', map_location=device))

vae = VAE(device)
vae.load_state_dict(torch.load(model_dir + 'vae.pth', map_location=device))

clf_params = parameters_to_vector(clf.parameters())
print(clf_params.shape)
