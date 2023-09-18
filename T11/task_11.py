import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision.io import read_image
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time as t
import copy
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.get_features = nn.Sequential( # Shape: [batch_size, 3, 224, 224]
            nn.Conv2d(3, 16, 3, 1), # Shape: [batch_size, 64, 222, 222]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # Shape: [batch_size, 64, 111, 111]

            nn.Conv2d(16, 32, 3, 1), # Shape: [batch_size, 64, 109, 109]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, 1), # Shape: [batch_size, 64, 107, 107]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # Shape: [batch_size, 64, 54, 54]

            nn.Conv2d(32, 32, 3, 1), # Shape: [batch_size, 32, 52, 52]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # Shape: [batch_size, 32, 26, 26]

            nn.Dropout(.25),
            nn.Flatten(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(20000, 10816),
            nn.BatchNorm1d(10816),
            nn.ReLU(),

            nn.Linear(10816, 10816),
            nn.BatchNorm1d(10816),
            nn.ReLU(),
            
            nn.Dropout(.2),
            
            nn.Linear(10816, 3)
        )

    def forward(self, x):
        return self.classifier(self.get_features(x))


def get_images(image_directory, resize=(224, 224)):
    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=f'{image_directory}/Train',
        transform=train_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=f'{image_directory}/Test',
        transform=test_transform
    )

    return train_dataset, test_dataset


def visualize_random_pictures(directory):
    fig = plt.figure(figsize=(10, 10))
    rows, columns = 5, 5
    train_dataset = torchvision.datasets.ImageFolder(root=f'{directory}/Train')
    rand_indeces = np.random.randint(low=0, high=len(train_dataset), size=(rows*columns,))

    images = []
    labels = []

    for i in rand_indeces:
        img, class_ = train_dataset[i]
        images.append(img)
        labels.append(train_dataset.classes[class_])

    # visualize these random images
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{labels[i-1]}")
    plt.show()


def train_my_cnn(batch_size, epochs, device):
    train_imgs, test_imgs = get_images('Dataset')
    train_loader = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_imgs, batch_size=batch_size, shuffle=False)

    model = CNN().to(device)

    dataloaders_dict = {
        "train": train_loader,
        "val": test_loader,
    }
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=.9)
    criterion = nn.CrossEntropyLoss()

    best_model, val_accs, val_losses, train_accs, train_losses = train(model, dataloaders_dict, criterion, optimizer, epochs, device)
    val_acc_history_cpu = [i.item() for i in val_accs]
    val_loss_history_cpu = val_losses
    train_acc_history_cpu = [i.item() for i in train_accs]
    train_loss_history_cpu = train_losses

    plot_names = ('MyCNN_AccuracyCurve', 'MyCNN_LossCurve')
    acc_loss_curves(train_acc_history_cpu, val_acc_history_cpu, train_loss_history_cpu, val_loss_history_cpu, names=plot_names)


def train_on_finetuning(batch_size, epochs, device):
    train_imgs, test_imgs = get_images('Dataset')
    num_classes = len(train_imgs.classes)
    train_loader = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_imgs, batch_size=batch_size, shuffle=False)

    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1),

        nn.Linear(num_ftrs, num_ftrs),
        nn.BatchNorm1d(num_ftrs),
        nn.ReLU(),
        nn.Dropout(.5),

        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    dataloaders_dict = {
        "train": train_loader,
        "val": test_loader,
    }
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=.9)
    criterion = nn.CrossEntropyLoss()

    best_model, val_accs, val_losses, train_accs, train_losses = train(model, dataloaders_dict, criterion, optimizer, epochs, device)
    val_acc_history_cpu = [i.item() for i in val_accs]
    val_loss_history_cpu = val_losses
    train_acc_history_cpu = [i.item() for i in train_accs]
    train_loss_history_cpu = train_losses

    plot_names = ('FinetunedAccuracyCurve', 'FinetunedLossCurve')
    acc_loss_curves(train_acc_history_cpu, val_acc_history_cpu, train_loss_history_cpu, val_loss_history_cpu, names=plot_names)


def train(model, dataloaders, criterion, optimizer, n_epochs, device):
    since = t.time()

    train_acc_history = []
    train_loss_history = []

    val_acc_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    y_pred_proba = torch.softmax(outputs, dim=1)
                    y_pred = y_pred_proba.argmax(dim=1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print(f"{preds = }")
                # print(f"{labels.data = }")
                running_corrects += torch.sum(y_pred == labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = t.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # get_model_statistics(model, 'Dataset', device)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history


def get_model_statistics(model, dataset_path, device):
    _, test_data = get_images(dataset_path)
    test_data_loader = DataLoader(test_data, batch_size=10)

    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X, y in test_data_loader:
            X = X.to(device)
            y = y.to(device)
            
            outputs = model(X)
            y_pred += torch.softmax(outputs, dim=1).argmax(dim=1).to('cpu').tolist()
            y_true += y.to('cpu').tolist()

    y_pred = [test_data.classes[i] for i in y_pred]
    y_true = [test_data.classes[i] for i in y_true]
    print(classification_report(y_true, y_pred))

    matrix = confusion_matrix(y_true, y_pred)
    
    cm_display = ConfusionMatrixDisplay(matrix, display_labels=test_data.classes)
    cm_display.plot()
    plt.savefig('confusion_matrix.png')


def acc_loss_curves(train_accs, test_accs, train_losses, test_losses, names=('AccuracyCurve', 'LossCurve')):
    fig1 = plt.figure()
    plt.plot(test_accs, label="Validation accuracies")
    plt.plot(train_accs, label="Train accuracies")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f'{names[0]}.png')

    fig2 = plt.figure()
    plt.plot(np.array(test_losses), label="Validation losses")
    plt.plot(np.array(train_losses), label="Train losses")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()    
    plt.savefig(f'{names[1]}.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Active device is: {device}')
    
    # train_on_finetuning(batch_size=32, epochs=10, device=device)
    train_my_cnn(batch_size=8, epochs=50, device=device)


if __name__ == '__main__':
    main()