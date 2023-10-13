from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import datasets, transforms


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


class PyTeen(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequential layers (fully connected/linear layers)
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512), # Input layer
            nn.ReLU(), # Activation func - input layer
            nn.Linear(512, 512), # Hidden layer
            nn.ReLU(), # Activation func - hidden layer
            nn.Linear(512, 10) # Output layer
        )
        # Loss/Cost func
        self.loss = nn.CrossEntropyLoss()
        # Optimizer - for adjusting weights
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, input):
        return self.layers(input)

    def train(self, feature, label):
        # Zero out all gradients at each step
        self.optimizer.zero_grad()
        # Make prediction using the NN
        pred = self.forward(feature) # Feature = input
        # Calcuate the loss using our defined loss func
        loss = self.loss(pred, label) # Label = actual
        # Put gradients in the .grad props
        loss.backward()
        # Update gradients
        self.optimizer.step()
        return loss.item() # Don't NEED to return loss - nice for training loop


# Data loaders for train/test data
BATCH_SIZE = 32
EPOCHS = 3

pyteen = PyTeen()

train_loader = dataloader.DataLoader(
    training_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

test_loader = dataloader.DataLoader(
    testing_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

# Training loop
for i in range(EPOCHS):
    total_loss = 0
    # Call train repeatedly with traiing data pairs (feature, label)
    for feature,label in train_loader:
        total_loss += pyteen.train(feature, label)
    print(f"Total loss for Epoch {i+1}: {total_loss/len(train_loader)}")
