import torch
import torch.optim as optim
from torch import nn

from dataloader import load_data
from models import SimpleHalMoonNN


def train(epochs, train_dataloader, optimizer, model):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):  # loop over the dataset multiple times

        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()

        print(f'[{epoch + 1:3d}] loss: {running_loss / 2:.3f}, train acc: {100 * correct // total} %')


    print('Finished Training')


def test(test_dataloader, model):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on {total} test: {100 * correct // total} %')


# *******************************************************

epochs = 500

x_dim = 2
y_dim = 2       # Number of classes
h_dim = [8, 16, 8, 4]


train_dataloader, test_dataloader = load_data('HalfMoon', batch_size=256)
model = SimpleHalMoonNN(x_dim, y_dim, h_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

train(epochs, train_dataloader, optimizer, model)
test(test_dataloader, model)

# this is how the model parameters can be saved:
# torch.save(model.state_dict(), './pretrained/model_reference.pth')

