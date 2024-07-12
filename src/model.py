import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Updated to 64 * 14 * 14
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the batch size
batch_size = 500

# Load the CIFAR-10 Dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

# Define Training and Evaluation Functions
def train(model, trainloader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def evaluate(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= len(testloader)
    accuracy = 100. * correct / len(testloader.dataset)
    print(f'Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return test_loss

# Initialize the criterion (loss function)
criterion = nn.CrossEntropyLoss()

# Define Learning Rate Schedulers
schedulers = {
    # Simple Schedulers
    #"LambdaLR": lambda optimizer: lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.75 ** epoch),
    #"LinearLR": lambda optimizer: lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=5),
    #"ConstantLR": lambda optimizer: lr_scheduler.ConstantLR(optimizer, factor=.50, total_iters=20),
    #"MultiplicativeLR": lambda optimizer: lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95),
    #"PolynomialLR": lambda optimizer: lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=2.0),
    #"ExponentialLR": lambda optimizer: lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
    #"ReduceLROnPlateau": lambda optimizer: lr_scheduler.ReduceLROnPlateau(optimizer, 'min'),

    # Step-wise Schedulers
    #"StepLR": lambda optimizer: lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1),
    #"MultiStepLR": lambda optimizer: lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1),
    #"SequentialLR": lambda optimizer: lr_scheduler.SequentialLR(optimizer, schedulers=[
    #    lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=5),
    #    lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #], milestones=[5]),
    #"ChainedScheduler": lambda optimizer: lr_scheduler.ChainedScheduler([
    #    lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=5),
    #    lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #]),

    # Cyclical Schedulers
    #"OneCycleLR": lambda optimizer: lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=50),
    #"CyclicLR": lambda optimizer: lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=2.5),
    #"CosineAnnealingLR": lambda optimizer: lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0),
    #"CosineAnnealingWarmRestarts": lambda optimizer: lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
}

# Train and evaluate the model with each scheduler
for scheduler_name, scheduler_fn in schedulers.items():
    print(f"Training with {scheduler_name} scheduler")
    
    # Initialize model, optimizer, and scheduler
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = scheduler_fn(optimizer)

    # Track learning rates
    learning_rates = []

    num_epochs = 20

    for epoch in range(num_epochs):  # Number of epochs can be adjusted
        train(model, trainloader, criterion, optimizer, epoch, device)
        val_loss = evaluate(model, testloader, criterion, device)
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)  # Pass the validation loss to the scheduler
        else:
            scheduler.step()
        
        # Record the learning rate
        learning_rates.append(optimizer.param_groups[0]['lr'])

    # Plot and save the learning rate graph
    plt.figure()
    plt.plot(range(1, len(learning_rates) + 1), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule - {scheduler_name}')
    plt.savefig(f'../visualizations/learning_rate_{scheduler_name}.png')
    plt.close()
