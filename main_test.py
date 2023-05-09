# %%
### Set up imports ###
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
import model_flower
import model_train
import model_test
import os
import datetime
import time

### Set up variables ###
tr_batchsize = 16  # The size of the training batches
val_test_batchsize = 16  # The size of the validation / testing batches
epochs = 20  # The number of epochs to do
validate_steps = 750  # The number of steps to complete before validation
learning_rate = 0.0001  # The learning rate to start at
load_model = True  # If a model should be requested to be loaded, or not
save_model = True  # If the model should be saved after testing, or not

# %%
# By default, set to use the CPU
deviceFlag = torch.device('cpu')

# If a GPU is available, use it
if torch.cuda.is_available():
    print(f'Found {torch.cuda.device_count()} GPUs.')
    deviceFlag = torch.device('cuda:0')  # Default to cuda 0, but can be changed.

print(f'Now the device is set to {deviceFlag}')

# %% md
### Data Loading and Transformations ###

testing_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # RGB mean & std estied on ImageNet
                         [0.229, 0.224, 0.225])
])

# Load the datasets of the Flower102 images
test_dataset = datasets.Flowers102(root='./dataset', split='test', transform=testing_transforms, download=True)

# Create the loaders for the datasets, to be used to train, validate and test the model
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=val_test_batchsize)

# %%
loaded_file = False

print("Creating Model...")
model = model_flower.FlowerModel()
print("Model created. Moving the Model to " + deviceFlag.type + "...")
model.to(deviceFlag)
print("Moved the Model to " + deviceFlag.type + ".")

# %%
### Criterion, Optimizer and Scheduler ###

# Negative Log Likelihood Loss
# criterion = nn.NLLLoss()

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# optimizer 1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# optimizer 2
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = 0.005, momentum = 0.9)

# Scheduler
scheduler = lr_scheduler.StepLR(optimizer, 500, 0.99)


# %%
# If going to load a model
if load_model:
    new_file_path = input("Input Model to load >> ").strip()
    # Ignore it if it's an empty string, and then try to load the file
    if new_file_path != "":
        if model_flower.load_state_model("models/" + new_file_path, model, criterion, optimizer, scheduler):
            print("\nUsing model file from " + new_file_path)
        else:
            print("Model failed to load, quitting...")
            quit()
    else:
        print("No file set to load. Using default Model.")

print()


# %%
### Test the Model ###
model_test.test_accuracy(model, test_loader, device_flag=deviceFlag)
