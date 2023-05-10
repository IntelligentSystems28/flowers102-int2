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

### Set up variables ###
tr_batchsize = 16  # The size of the training batches
val_test_batchsize = 16  # The size of the validation / testing batches
epochs = 600  # The number of epochs to do
# This is set down below, to go off every 10 epochs
validate_steps = 1600  # The number of steps to complete before validation
learning_rate = 0.0001  # The learning rate to start at for a new model
load_model = False  # If a model should be requested to be loaded, or not

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

training_transforms = transforms.Compose([
    # Randomly rotate it 90 degrees
    transforms.RandomRotation(180),
    # Randomly do an auto contrast on the image
    transforms.RandomAutocontrast(),
    # Randomly sharpen the image
    transforms.RandomAdjustSharpness(1.5, 0.5),
    # Randomly crop an area of the flower of size 224x224
    transforms.RandomResizedCrop(224),
    # Flip it horizontally, or don't
    transforms.RandomHorizontalFlip(),
    # Flip it vertically, or don't
    transforms.RandomVerticalFlip(),
    # Convert the image to a Tensor
    transforms.ToTensor(),
    # Normalize the Tensor values so that they're easier for the
    # model to train from
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    # Randomly erase an area from an image
    transforms.RandomErasing()
])

validation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # RGB mean & std estied on ImageNet
                         [0.229, 0.224, 0.225])
])

testing_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # RGB mean & std estied on ImageNet
                         [0.229, 0.224, 0.225])
])

# Load the datasets of the Flower102 images
train_dataset = datasets.Flowers102(root='./dataset', split='train', transform=training_transforms, download=True)
valid_dataset = datasets.Flowers102(root='./dataset', split='val', transform=validation_transforms, download=True)
test_dataset = datasets.Flowers102(root='./dataset', split='test', transform=testing_transforms, download=True)

# Create the loaders for the datasets, to be used to train, validate and test the model
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=tr_batchsize,
                                           shuffle=True)

validate_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=val_test_batchsize)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=val_test_batchsize)

# %%
print("Creating Model...")
model = model_flower.FlowerModel()
print("Model created. Moving the Model to " + deviceFlag.type + "...")
model.to(deviceFlag)
print("Moved the Model to " + deviceFlag.type + ".")

### Create the Criterion and Optimizer ###

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

### Create the Scheduler ###
# A step scheduler that decreases the learning rate every 500
# steps by x0.99
scheduler = lr_scheduler.StepLR(optimizer, 500, 0.99)


### Model Loading ###

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


### Train the Model ###
# Attempt 1 - 12722.4395 seconds, 53% accuracy on the Test Data
# Attempt 2 - 10543.1400 seconds, 52.8% accuracy on the Test Data
# Attempt 3 - 10359.2405 seconds, 54.4% accuracy on the Test Data

model_name = "flower-model"
model_save_location = f"{model_name}.pt"
torch.save(model, model_save_location)
model_train.train_classifier(model, train_loader, validate_loader, optimizer, criterion,
                             optim_scheduler=scheduler, device_flag=deviceFlag, epochs=epochs,
                             validate_steps=validate_steps, validate_stepped=True,
                             validate_epoch=False, validate_end=True
                             )

print("\n------------------------\n")


# Save the final model
# new_file_path = model_flower.save_model(model, criterion, optimizer, scheduler, name=model_name)
state_save_location = f"{model_name}-state.pt"
model_flower.save_state_model(model, criterion, optimizer, scheduler, state_save_location)
print(f"Final model saved to {state_save_location}.")


### Test the Model ###
print("\n")
model_test.test_accuracy(model, test_loader, device_flag=deviceFlag)
