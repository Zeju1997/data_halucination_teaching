import torch
import torch.nn.functional as F

from dataloader import load_data
from models import VAE_HalfMoon, cVAE_HalfMoon

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap




num_epochs = 600
batch_size = 128
learning_rate = 1e-3
use_gpu = True

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print(device)
# vae = VAE_HalfMoon(device)
vae = cVAE_HalfMoon(device)
vae = vae.to(device)


train_dataloader, test_dataloader = load_data('HalfMoon', batch_size=batch_size)

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




vae.eval()
with torch.no_grad():
    # For p(x,y):
    # X, y_logits = vae.sample(num=1000)
    # For p(x|y):
    y = torch.tensor([0,0,0,0,0, 1,1,1,1,1], dtype=torch.int64).to(device)
    y_oh = F.one_hot(y)
    X = vae.sample(y_oh, num=10)

X = X.data.cpu().numpy()
# y = torch.argmax(y_logits, dim=1).data.cpu().numpy()  # For p(x,y)

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

fig, ax = plt.subplots()
ax.set_title("Input data")

ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
            edgecolors='k')

plt.tight_layout()
plt.show()


# this is how the VAE parameters can be saved:
# torch.save(vae.state_dict(), './pretrained/vae.pth')