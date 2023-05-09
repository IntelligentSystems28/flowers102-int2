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
import time

### Set up variables ###
tr_batchsize = 16  # The size of the training batches
val_test_batchsize = 16  # The size of the validation / testing batches
epochs = 10  # The number of epochs to do
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
loaded_file = False

print("Creating Model...")
model = model_flower.FlowerModel()
print("Model created. Moving the Model to " + deviceFlag.type + "...")
model.to(deviceFlag)
print("Moved the Model to " + deviceFlag.type + ".")



# %%
# Negative Log Likelihood Loss
# criterion = nn.NLLLoss()

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# optimizer 1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# optimizer 2
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = 0.005, momentum = 0.9)


# %%
# Scheduler
scheduler = lr_scheduler.StepLR(optimizer, 500, 0.99)


### Model Loading ###

# If going to load a model
if load_model:
    new_file_path = input("Input Model to load >> ").strip()
    # Ignore it if it's an empty string, and then try to load the file
    if new_file_path != "":
        if model_flower.load_model("models/" + new_file_path, model, criterion, optimizer, scheduler):
            print("\nUsing model file from " + new_file_path)
        else:
            print("Model failed to load, quitting...")
            quit()
    else:
        print("No file set to load. Using default Model.")

print()

# %%
### Train the Model ###

train_time_start = time.time()
train_time_end = train_time_start + 60 * 60 * 6  # Last number is number of hours of training.

train_save_time_gap = 60 * 30  # Save every half hour (when the training for a set of epochs is done)
train_save_time_next = train_time_start + train_save_time_gap  # Set the first save time
train_save_max = 5  # The maximum number of models to save
train_saves = []

model_name = "model"
batches_done = 0

train_iteration = 0
train_iter_time_total = 0

while True:
    time_before = time.time()
    print("Iteration [{}] of training. {:.3f}s have passed since the start.".format(train_iteration + 1,
                                                                                    time_before - train_time_start))
    if train_iteration > 0:
        print("{} Batches complete total. Average of {:.3f}s per iteration.".format(
            batches_done, train_iter_time_total / train_iteration)
        )
    print()

    ### TRAIN THE MODEL###
    batches_done += model_train.train_classifier(model, train_loader, validate_loader, optimizer, criterion,
                                                 optim_scheduler=scheduler, device_flag=deviceFlag, epochs=epochs,
                                                 validate_steps=validate_steps, validate_stepped=False,
                                                 validate_epoch=False, validate_end=True,
                                                 end_time=train_time_end,
                                                 epochs_start=epochs*train_iteration, batches_start=batches_done
                                                 )
    print()

    time_after = time.time()
    # Check if there's no time left
    if time_after >= train_time_end:
        break

    train_iter_time_total += time_after - time_before

    # If saving the model...
    if save_model and time_after >= train_save_time_next:
        # Save the model
        new_file_path = model_flower.save_model(model, criterion, optimizer, scheduler, name=model_name)
        print(f"New model saved to {new_file_path}")
        train_saves.append(new_file_path)
        # If the size is now over the limit, delete the file of the first index
        if len(train_saves) > train_save_max:
            file_path = train_saves.pop(0)
            if os.path.exists(file_path):
                os.remove(file_path)
        # And then update the next time to save
        train_save_time_next += train_save_time_gap

    train_iteration += 1

# Save the final model
new_file_path = model_flower.save_model(model, criterion, optimizer, scheduler, name=model_name)
print(f"Final model saved to {new_file_path}.")

# %%
### Test the Model ###
model_test.test_accuracy(model, test_loader, device_flag=deviceFlag)
