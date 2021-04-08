# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from pathlib import Path
from PIL import Image
import os, sys

data_dir = Path("data/")

class CNN:
    def __init__(self,isnew = False):
        if isnew == True:
            self.model = models.resnet50(pretrained=True) #torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
        else:
            self.model = models.resnet50()
            self.model.load_state_dict(torch.load('KEYVIBE_model.pth'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses, self.test_losses = [], []

    def load_split_train_test(self, data_dir, valid_size=.2):
        train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.ToTensor(),
                                               ])

        test_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                              transforms.ToTensor(),
                                              ])

        train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
        test_data = datasets.ImageFolder(data_dir, transform=test_transforms)


        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)

        from torch.utils.data.sampler import SubsetRandomSampler

        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        self.trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
        self.testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)

    def train(self):
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(nn.Linear(2048, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(512, 2),
                                      nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=0.003)
        self.model.to(self.device)

        epochs = 1
        steps = 0
        running_loss = 0
        print_every = 10

        for epoch in range(epochs):
            for inputs, labels in self.trainloader:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.testloader:
                            inputs, labels = inputs.to(self.device),labels.to(self.device)
                            logps = self.model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    self.train_losses.append(running_loss / len(self.trainloader))
                    self.test_losses.append(test_loss / len(self.testloader))
                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Test loss: {test_loss / len(self.testloader):.3f}.. "
                          f"Test accuracy: {accuracy / len(self.testloader):.3f}")
                    running_loss = 0
                    self.model.train()
        torch.save(self.model, 'KEYVIBE_model.pth')

    def plot_loss(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()

    def resize(self, path):
        # This was in case the reszing in the load function did not work
        dirs = os.listdir(path)
        for item in dirs:
            if os.path.isfile(path + item):
                im = Image.open(path + item)
                f, e = os.path.splitext(path + item)
                imResize = im.resize((200, 200), Image.ANTIALIAS)
                imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

    def RUN(self):
        input_image = Image.open("./IMG_7740.JPG")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        with torch.no_grad():
            output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities)

def main():
    Resnet = CNN(True)
    Resnet.load_split_train_test(data_dir, .2)
    Resnet.train()
    Resnet.plot_loss()
    Resnet.RUN()


if __name__ == '__main__':
    main()