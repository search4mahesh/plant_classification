#Imaport libraries
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Hyper parameters
num_epochs = 10
num_classes = 12
batch_size = 32
learning_rate = 0.0001

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#Specify transforms using torchvision library

transformation = transforms.Compose([
    transforms.Resize(size=(299, 299)),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# load each dataset and apply transformations
train_set = datasets.ImageFolder("<USER_HOME>/machine-learning/datasets/Plant_Seedlings_Classification/all/train/train", transform=transformation)

valid_set = datasets.ImageFolder("<USER_HOME>/machine-learning/datasets/Plant_Seedlings_Classification/all/train/valid", transform=transformation)

#Put into Dataloader 

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=batch_size, shuffle=True)

print(len(train_loader.dataset))
print(len(valid_loader.dataset))

class_to_idx = train_set.class_to_idx
print(class_to_idx)

model = models.inception_v3(pretrained=True)
num_ftrs = model.fc.in_features

model.fc1 = nn.Linear(num_ftrs, 512)
model.relu1 = nn.ReLU()
model.dropout1 = nn.Dropout(0.5)

model.fc2 = nn.Linear(512, num_classes)
model.softmax = nn.Softmax()
model.dropout2 = nn.Dropout(0.5)



model = model.cuda()


#set error function using torch.nn as nn lib
criterion = nn.CrossEntropyLoss()
#set the optimiser function using torch.optim as optim library
optimizer = optim.Adam(model.parameters(), lr= learning_rate)


#Training
min_loss = 1000
pre_best_min_loss = 10000
total_step = len(train_loader)
for epoch in range(num_epochs):
    print("Starting epoch :" + str(epoch))
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs, aux = model(images)
        loss = criterion(outputs, labels)

        #Backward and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
        print('Step [{}/{}], Loss: {:.4f}'
            .format(i+1, total_step, min_loss))
    #save model with lowest loss
    if min_loss < pre_best_min_loss:
        pre_best_min_loss = min_loss
        print("saving best model with loss " + str(min_loss))
        torch.save(model, 'plant_classification_inceptionv3_model_v3_more_layers.pt')

        #if (i+1) % 100 == 0:
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch+1, num_epochs, i+1, total_step, min_loss))

#Test the model
model.eval()
with torch.no_grad():
    correct  = 0
    total = 0
    for i, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs, aux = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))





        





