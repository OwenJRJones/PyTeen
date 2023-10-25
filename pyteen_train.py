"""Training PyTeen Network"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import pyteen


BATCH_SIZE = 16
EPOCHS = 4
DEVICE = DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Transform PIL image to tensor of image dimensions
cust_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Lambda(lambda img: img.reshape(28*28))])

# Load training/testing data
training_data = datasets.MNIST(
    root="./data", # Define where to store/look for data
    download=True, # Downlaod data if not already
    train=True, # Set true for training data
    transform=cust_transform) # Use custom transform

testing_data = datasets.MNIST(
    root="./data",
    download=True,
    train=False,
    transform=cust_transform)

# Run this puppy!
pyteen = pyteen.PyTeen()
pyteen.to(DEVICE)

# Data loaders for train/test data
train_loader = DataLoader(
    training_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

# Training loop
print("-----------------Training Network-----------------")

for i in range(EPOCHS):
    total_loss = 0
    # Call train repeatedly with traiing data pairs (feature, label)
    for feature,label in tqdm(train_loader):
        feature = feature.to(DEVICE)
        label = label.to(DEVICE)
        total_loss += pyteen.train_net(feature, label)
    print(f"Total loss for Epoch {i+1}: {total_loss/len(train_loader)}")

# Save trained network weights
torch.save(pyteen.state_dict(), "py_teen.pth")

print("\033[1;32mTrained state saved.")
