from models.resnet import *
from utils import *


from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.nn as nn
import torch.optim as optim


PAD = 4
EPOCHS = 20
LR  = 0.001
MOMENTUM= 0.8



classes = ["airplane", "automobile", "bird", "cat", "deer", 
           "dog", "frog", "horse", "ship", "truck"]


print(f"----------Compute image mean & std----------")
trainloader = get_train_loader(transform=None)
mean, std = get_stats(trainloader)


denorm = UnNormalize(mean, std) 

print(f"----------Normailizing Images----------")
train_transform = get_train_transform(mean, std)
test_transform = get_test_transform(mean, std)

print(f"----------Load and Transform Images----------")
trainloader = get_train_loader(transform=train_transform)
testloader = get_test_loader(transform=test_transform)


device = get_device()
print(f"----------Device type - {device}----------")


print(f"----------Model Summary----------")
model = ResNet18().to(device)
get_summary(model, device)


model =  ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
scheduler = OneCycleLR(optimizer, max_lr=0.05, epochs=EPOCHS, steps_per_epoch=len(trainloader))

print(f"----------Training Model----------")
results = train_model(model, criterion, device, trainloader, testloader, optimizer, scheduler, EPOCHS)

print(f"----------Loss & Accuracy Plots----------")
make_plot(results)


max_images = 128
test_images = [x[0] for x in testloader.dataset]
test_images = torch.stack(test_images[:max_images])
test_targets = torch.tensor(testloader.dataset.targets[:max_images]).to(device)
print(f"----------Inference on {max_images} test images----------")


test_predictions = model(test_images.to(device))

miss_index, hit_index = get_idxs(test_predictions, test_targets, device)


print(f"----------missclassifid index length is {len(miss_index)}----------")
print(f"----------Correctly classified index length is {len(hit_index)}----------")


print(f"----------Visualize model predictions----------")
show_images_pred(test_images, test_targets, test_predictions, denorm)


test_images = test_images.to(device)

print(f"----------Visualize misclassified predictions----------")
show_images_pred(test_images[miss_index], test_targets[miss_index], test_predictions[miss_index], denorm)


print(f"----------Generate heatmaps for test images----------")
heatmaps = gradcam_heatmap(model, test_predictions, test_images, device)

heatmaps = heatmaps.to(device)

hit_maps = heatmaps[hit_index]
miss_maps = heatmaps[miss_index]

print(f"----------isualize misclassified GRADCAM----------")
show_images_cam(test_images, test_targets, test_predictions, heatmaps, miss_index, denorm)


print(f"----------Visualize correctly classified GRADCAM----------")
show_images_cam(test_images, test_targets, test_predictions, heatmaps, hit_index, denorm)
