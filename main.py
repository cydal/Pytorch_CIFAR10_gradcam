import resnet
import utils



PAD = 4
EPOCHS = 300
LR  = 0.001
MOMENTUM= 0.8

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
SCHEDULER = OneCycleLR(optimizer, max_lr=0.05, epochs=EPOCHS, steps_per_epoch=len(trainloader))



trainloader = get_train_loader(transform=None)

mean, std = get_stats(trainloader)

denorm = UnNormalize(mean, std) 


train_transform = get_train_transform(mean, std)
test_transform = get_test_transform(mean, std)

trainloader = get_train_loader(transform=train_transform)
testloader = get_test_loader(transform=test_transform)


device = get_device()


model = ResNet18().to(device)
get_summary(model, device)




model =  ResNet18().to(device)


results = train_model(model, criterion, device, trainloader, testloader, optimizer, scheduler, EPOCHS)

