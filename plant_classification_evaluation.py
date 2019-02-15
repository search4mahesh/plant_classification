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
from sklearn.metrics import confusion_matrix

# Hyper parameters
num_epochs = 35
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

valid_set = datasets.ImageFolder(
    "<USER>/machine-learning/datasets/Plant_Seedlings_Classification/all/train/test", transform=transformation)

classes = valid_set.classes

#Put into Dataloader

valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=batch_size, shuffle=True)

print(len(valid_loader.dataset))

model = torch.load('plant_classification_inceptionv3_model_v3_more_layers.pt')


#print(model)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    predicted_total = []
    labels_total = []
    fig = plt.figure(figsize=(10, 10))
    for i, (images, labels) in enumerate(valid_loader):
        sub = fig.add_subplot(1, 5, i+1)
        images = images[0:5].to(device)
        labels = labels[0:5].to(device)
        outputs = model(images)
        _ , predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_total += predicted.cpu().numpy().tolist()
        labels_total += labels.cpu().numpy().tolist()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
       100 * correct / total))

    conf_matrix = confusion_matrix(
        labels_total, predicted_total)
    print(conf_matrix)
