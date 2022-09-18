# %%
import torch
import torch.nn.functional as F
import torchvision.utils

from dataloader import load_data
from models import VAE_MNIST

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.ion()




num_epochs = 500
batch_size = 128
learning_rate = 1e-3
use_gpu = True

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print(device)
vae = VAE_MNIST(device)
vae = vae.to(device)


train_dataloader, test_dataloader = load_data('MNIST', batch_size=batch_size)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

# set to training mode
vae.train()

train_loss_avg = []

print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    
    for x_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        
        y_batch = F.one_hot(y_batch, num_classes=2).type(torch.FloatTensor) * 2. - 1
        y_batch = y_batch.to(device)

        x_batch = x_batch.to(device)

        loss,_ = vae(x_batch, y_batch)
        
        # backpropagation
        loss.backward()
        
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        
        train_loss_avg[-1] += loss.item()
        num_batches += 1
        
    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average negative ELBO: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))




# %%
vae.eval()

# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img, title):
    img = to_img(img)
    npimg = img.numpy()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, labels, model, title):

    with torch.no_grad():
        labels = F.one_hot(labels, num_classes=2).type(torch.FloatTensor)
        labels = labels.to(device)
        images = images.to(device)
        images, y_logits = vae.reconstruction(images, labels)
        images = images.cpu()
        y = torch.argmax(y_logits, dim=1).data.cpu().numpy()
        print("Recons:")
        print(y[0:10])
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[0:10], 10, 5).numpy()
        plt.title(title)
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

images, labels = iter(test_dataloader).next()

# First visualise the original images
show_image(torchvision.utils.make_grid(images[0:10],10,5), "Original")
plt.show()

# Reconstruct and visualise the images using the vae
visualise_output(images, labels, vae, "Reconstruction")


# Samples
with torch.no_grad():

    # sample images
    img_samples, y_logits = vae.sample()
    y = torch.argmax(y_logits, dim=1).data.cpu().numpy()
    print("Samples:")
    print(y)
    img_samples = img_samples.cpu()

    fig, ax = plt.subplots(figsize=(5, 5))
    show_image(torchvision.utils.make_grid(img_samples,10,5), "Samples")
    plt.show()

# %%
# this is how the VAE parameters can be saved:
# torch.save(vae.state_dict(), './pretrained/mivae.pth')

