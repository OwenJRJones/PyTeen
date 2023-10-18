"""Testing PyTeen Network"""

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pyteen


BATCH_SIZE = 16

# Transform PIL image to tensor of image dimensions
cust_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Lambda(lambda img: img.reshape(28*28))])

# Load testing data
testing_data = datasets.MNIST(
    root="./data",
    download=True,
    train=False,
    transform=cust_transform)

# Build test loader
test_loader = DataLoader(
    testing_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

pyteen = pyteen.PyTeen()

# load saved weights into network
saved_paramters = torch.load('py_teen.pth')
pyteen.load_state_dict(saved_paramters)

# Evaluation/Testing loop
print("-----------------Testing Network-----------------")

num_correct = 0

for feature,label in tqdm(test_loader):
    pred = pyteen.predict(feature)
    num_correct += (pred == label).sum()

accuracy = ((num_correct / (len(test_loader) * BATCH_SIZE)).item()) * 100
print(f"Accuracy: {accuracy:.4f}%")

print("\033[1;32mTesting complete.")
