import torch
import torch.nn as nn
import torchvision.datasets as datasets


def validation(model, validate_loader, val_criterion, device_flag="cpu"):
    val_loss_running = 0
    acc = 0

    model.eval()

    # a dataloader object is a generator of batches, each batch contain image & label separately
    for images, labels in iter(validate_loader):
        # Send the data onto choosen device
        images = images.to(device_flag)
        labels = labels.to(device_flag)

        output = model(images)

        # *images.size(0) # .item() to get a scalar in Torch.tensor out
        val_loss_running += val_criterion(output, labels).item()

        probabilities = torch.exp(output)  # as in the model we use the .LogSoftmax() output layer

        equality = (labels.data == probabilities.max(dim=1)[1])
        acc += equality.type(torch.FloatTensor).mean()

    return val_loss_running, acc


def test_accuracy(model, test_loader, device_flag="cpu"):

    # Do validation on the test set
    model.eval()
    model.to(device_flag)

    with torch.no_grad():

        accuracy = 0
        predicted_correctly = 0
        images_checked = 0
        images_to_check = len(test_loader.dataset)

        for images, labels in iter(test_loader):
            print("\r{}/{} - {:.3f}% complete".format(images_checked, images_to_check,
                                                      (images_checked / images_to_check)*100),end="")

            images, labels = images.to(device_flag), labels.to(device_flag)

            output = model.forward(images)

            probabilities = torch.exp(output)

            equality = (labels.data == probabilities.max(dim=1)[1])

            predicted_correctly += torch.sum(equality)

            accuracy += equality.type(torch.FloatTensor).mean()

            images_checked += images.size(0)

        print("\rTest Accuracy: {},".format(accuracy/len(test_loader)),
              "{} / {} correctly predicted.".format(predicted_correctly, len(test_loader.dataset)))