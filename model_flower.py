import torch
import datetime
import os
import torch.nn as nn


class FlowerModel(nn.Module):
    def __init__(self, num_classes=102):
        super(FlowerModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            # first linear must be image size * image size * last Conv2d out channel
            nn.Linear(14 * 14 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.layer1(x)

        out = self.fc(out)
        return out


def save_model(model, criterion, optimizer, scheduler, name="model"):
    save_data = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }

    file_path = f"models/{name}-{str(datetime.datetime.now()).replace(':', '-')}.pt"
    torch.save(save_data, file_path)
    return file_path


def load_model(file_path: str, model: nn.Module, criterion, optimizer, scheduler):
    # Try to load the file, only if it exists
    if os.path.exists(file_path):
        # Try to load everything
        file_data = torch.load(file_path)

        # If it did successfully, load them to the model and return true
        model.load_state_dict(file_data["model"])
        criterion.load_state_dict(file_data["criterion"])
        optimizer.load_state_dict(file_data["optimizer"])
        scheduler.load_state_dict(file_data["scheduler"])
        return True
    return False
