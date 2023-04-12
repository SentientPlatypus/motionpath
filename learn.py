import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# define a simple neural network with one hidden layer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# create an instance of the network
net = Net()

# load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# train the network
batch_size = 64
num_epochs = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # flatten the images into a batch of vectors
        images = images.reshape(-1, 28*28)

        # zero out the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = net.forward(images)

        # compute the loss
        loss = criterion(outputs, labels)

        # backward pass and parameter update
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# evaluate the network on the test set
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # flatten the images into a batch of vectors
        images = images.reshape(-1, 28*28)

        # forward pass
        outputs = net.forward(images)

        # compute the predicted class
        _, predicted = torch.max(outputs.data, 1)

        # count the number of correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {(100 * correct / total):.2f}%')
