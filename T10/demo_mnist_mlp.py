"""
Dataset references:
http://pjreddie.com/media/files/mnist_train.csv
http://pjreddie.com/media/files/mnist_test.csv
"""


from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


torch.manual_seed(1337)


class MnistMlp(torch.nn.Module):
    
    def __init__(self, inputnodes: int, hiddennodes: int, outputnodes: int) -> None:
        super().__init__()

        # number of nodes (neurons) in input, hidden, and output layer
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=inputnodes, out_features=hiddennodes),
            torch.nn.BatchNorm1d(num_features=hiddennodes),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=hiddennodes, out_features=hiddennodes),
            torch.nn.BatchNorm1d(num_features=hiddennodes),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=hiddennodes, out_features=outputnodes),
            torch.nn.Dropout(p=.5)
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MnistDataset(Dataset):
    
    def __init__(self, filepath: Path) -> None:
        super().__init__()

        self.data_list = None
        with open(filepath, "r") as f:
            self.data_list = f.readlines()

        # conver string data to torch Tensor data type
        self.features = []
        self.targets = []
        for record in self.data_list:
            all_values = record.split(",")
            features = np.asfarray(all_values[1:])
            target = int(all_values[0])
            self.features.append(features)
            self.targets.append(target)

        self.features = torch.tensor(np.array(self.features), dtype=torch.float) / 255.0
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.long)
        # print(self.features.shape)
        # print(self.targets.shape)
        # print(self.features.max(), self.features.min())

    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]

def train_nn(path_to_model):
    # Device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NN architecture:
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate is 0.1
    learning_rate = 0.1
    # batch size
    batch_size = 12
    # number of epochs
    epochs = 10

    # Load mnist training and testing data CSV file into a datasets
    train_dataset = MnistDataset(filepath="./mnist_train.csv")
    test_dataset = MnistDataset(filepath="./mnist_test.csv")

    # Make data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Define NN
    model = MnistMlp(inputnodes=input_nodes, 
                     hiddennodes=hidden_nodes, 
                     outputnodes=output_nodes)
    # Number of parameters in the model
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device=device)
    
    # Define Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    logs = []

    ##### Training! #####
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (features, target) in enumerate(train_loader):
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(features), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        
        ##### Testing! #####
        model.eval()
        test_loss = 0
        correct = 0
        with torch.inference_mode():
            for features, target in test_loader:
                features, target = features.to(device), target.to(device)
                output = model(features)
                current_loss = criterion(output, target).item()
                test_loss += current_loss # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print('\nEpoch: {}\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            accuracy))
        logs.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy
        })

    print(logs)
    plt.title(f'Train/test loss curves, final accuracy: {logs[-1]["accuracy"]:.1f}%')
    plt.plot([log['epoch'] for log in logs], [log['train_loss'] for log in logs], label='Train loss')
    plt.plot([log['epoch'] for log in logs], [log['test_loss'] for log in logs], label='Test loss')
    plt.xticks([log['epoch'] for log in logs])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_test_loss.png')

    ##### Save Model! #####
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), path_to_model)


def test_nn(path_to_model):
    test_dataset = MnistDataset(filepath="./mnist_test.csv")
    X, y_true = test_dataset.features, test_dataset.targets

    model = MnistMlp(784, 200, 10)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    y_pred = model(X).argmax(dim=1, keepdim=True)
    matrix = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(matrix)
    print(f'\nThe total accuracy is: {accuracy_score(y_true, y_pred)}')
    print(f'The accuracies for each class:\n{matrix.diagonal()/matrix.sum(axis=1)}')
    print(f'\nThe total precision is: {precision_score(y_true, y_pred, average="macro")}')
    print(f'The precisions for each class are:\n{precision_score(y_true, y_pred, average=None)}')
    print(f'\nThe total recall is: {recall_score(y_true, y_pred, average="macro")}')
    print(f'The recalls for each class are:\n{recall_score(y_true, y_pred, average=None)}')
    print(f'\nThe total f1_score is: {f1_score(y_true, y_pred, average="macro")}')
    print(f'The f1_scores for each class are:\n{f1_score(y_true, y_pred, average=None)}')
    print(classification_report(y_true, y_pred, target_names=[str(x) for x in range(10)]))
    cm_display.plot()
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    # train_nn("mnist_001.pth")
    test_nn("mnist_001.pth")