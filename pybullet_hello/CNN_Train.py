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
import os

# data_dir = Path('/content/drive/My Drive/KEYVIBE/CNN_Testing/data/')

class CNN(nn.Module):
    def __init__(self, isnew = False):
        super(CNN, self).__init__()
        if isnew == True:
            self.model = models.resnet18(pretrained=True) #torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
        else:
            # self.model = models.resnet50(pretrained=False)
            # self.model = torch.load('CNN_models/KEYVIBE_model_best_nathan.pth', map_location=torch.device('cpu'))
            self.model = torch.load('CNN_models/180Traj_Natural_Resnet18.pth', map_location=torch.device('cpu'))
            if torch.cuda.is_available():
              self.model.cuda()
        with open("CNN_models/action.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        self.train_losses, self.test_losses = [], []

    def load_split_train_test(self, data_dir, valid_size=.2):
        train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.RandomVerticalFlip(),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               ])

        test_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.RandomVerticalFlip(),
                                               transforms.RandomHorizontalFlip(),
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
        num_ftrs = self.model.fc.in_features
        print(num_ftrs)
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                      nn.ReLU(),
                                      nn.Dropout(.2),
                                      nn.Linear(64, 6),
                                      nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss().to(self.device)
        optimizer = optim.Adam(self.model.fc.parameters(), lr=0.003)
        self.model.to(self.device)

        epochs = 15
        steps = 0
        running_loss = 0
        print_every = 5
        self.model.train()

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
        # This was in case the resizing in the load function did not work
        dirs = os.listdir(path)
        for item in dirs:
            if os.path.isfile(path + item):
                im = Image.open(path + item)
                f, e = os.path.splitext(path + item)
                imResize = im.resize((200, 200), Image.ANTIALIAS)
                imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

    def RUN(self, imgpath, live=False):
        self.model.eval()
        self.model.cpu()
        if live:
            input_image = Image.fromarray(imgpath)
        else:
            input_image = Image.open(imgpath)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        """ that was wrong """
        # preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        with torch.no_grad():
            output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        prob, catid = torch.topk(probabilities, 6)
        # for i in range(prob.size(0)):
        #     print(self.categories[catid[i]], prob[i].item())
        # print(probabilities)
        return probabilities


def main():
    Resnet = CNN(False).cpu()
    # Resnet.load_split_train_test(data_dir, .2)
    # Resnet.train()
    # Resnet.plot_loss()
    test_img = os.getcwd() + "/expert_trajectories/export/east/0p1.jpg"
    probs = Resnet.RUN(test_img)
    # probs = torch.tensor([0.365, 0.9055, 0.2201, 0.0580])
    action = int(np.argmax(probs))
    print(action)
    # id_to_action_name = {0: 'e', 1: 'n', 2: 's', 3: 'w'}  # old 4 actions
    id_to_action_name = {0: 'e', 1: 'g', 2: 'n', 3: 'r', 4: 's', 5: 'w'}  # new 6 actions
    print(id_to_action_name[action])


if __name__ == '__main__':
    main()

