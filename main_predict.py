import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import model_flower
import json

# By default, set to use the CPU
deviceFlag = torch.device('cpu')

# If a GPU is available, use it
if torch.cuda.is_available():
    print(f'Found {torch.cuda.device_count()} GPUs.')
    deviceFlag = torch.device('cuda:0')  # Default to cuda 0, but can be changed.

print(f'The device is set to {deviceFlag}')

testing_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # RGB mean & std estied on ImageNet
                         [0.229, 0.224, 0.225])
])

# Load the model
print("Loading the model...")
model = torch.load("flower-model.pt")
print("Model loaded.")
# Put it into eval mode
model.eval()

# Load the state of the entire model
print("Loading the Model's state...")
model_flower.load_state_model("models/flower-model-2023-05-09 22-10-23.281734.pt.pt", model)
print("Model state loaded.")
# Move it to the correct device
model.to(deviceFlag)

# Get the location of the image to predict
image_loc = "images/image_00405.jpg"
image_loc = "dataset/flowers-102/jpg/image_00181.jpg"

# Load the image and convert it into a pyTorch Tensor
print(f"Loading the '{image_loc}' Image...")
image = Image.open(image_loc)
image_tensor = testing_transforms(image).float()
image_tensor = image_tensor.unsqueeze(0)
np_image = np.array(image_tensor)
image_tensor = torch.from_numpy(np_image)
print("Image loaded.")

# Load the class names for the classes
print("Loading Class-Name mappings...")
with open("class_to_name.json") as file:
    class_data = json.load(file)
print("Class-Name mappings loaded.")

# Move it to the correct device
image_tensor = image_tensor.to(deviceFlag)

# Calculate the predicted class of the image
print("Predicting...")
with torch.no_grad():
    predicted = model(image_tensor)
print("Predicted.")
# Get the values from the predictions
probabilities = torch.topk(predicted, 5)[0]
labels = torch.topk(predicted, 5)[1]

# If the GPU is being used, move the values back to the CPU
if torch.cuda.is_available():
    probabilities = probabilities.cpu()
    labels = labels.cpu()

# Convert the numbers into numpy
probabilities = probabilities.detach().numpy()
labels = labels.detach().numpy()

print("The most likely flower for this to be is a {} with a {:.3f}% probability ".format(
    class_data[str(labels[0][0])], probabilities[0][0] * 100
))

print("The next 4 most likely flowers for this to be are:")
for i in range(1,len(probabilities[0])):
    print("{} with a {:.3f}% probability.".format(
        class_data[str(labels[0][i])],
        probabilities[0][i] * 100
    ))
