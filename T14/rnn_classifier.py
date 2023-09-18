from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchtext
from torch import nn
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_input, embedding_output, hidden_dim, n_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding_layer = nn.Embedding(num_embeddings=embedding_input, embedding_dim=embedding_output)
        self.lstm = nn.LSTM(input_size=embedding_output, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Sequential(
            nn.BatchNorm1d(num_features=2*hidden_dim),
            nn.Dropout(.1),
            nn.Linear(2*hidden_dim, 2)
        )

    def forward(self, X):
        hidden, carry = torch.randn(2*self.n_layers, len(X), self.hidden_dim).cuda(), torch.randn(2*self.n_layers, len(X), self.hidden_dim).cuda()
        lstm_output, (hidden, carry) = self.lstm(self.embedding_layer(X), (hidden, carry))
        output = self.linear(lstm_output[:, -1])
        
        return output, output.softmax(dim=1)


def get_imdb_dataset(device):
    train_dataset, test_dataset = torchtext.datasets.IMDB()
    nltk.download('stopwords')

    tokenizer = get_tokenizer("basic_english")
    def build_vocabulary(datasets):
        for dataset in datasets:
            for _, text in dataset:
                yield tokenizer(text)

    vocab = build_vocab_from_iterator(build_vocabulary([train_dataset, test_dataset]), min_freq=1, specials=["<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])

    max_words = 25

    def vectorize_batch(batch):
        Y, X = list(zip(*batch))
        Y = [0 if y == 1 else 1 for y in Y]
        X = [vocab(tokenizer(text)) for text in X]
        X = [[token for token in tokens if token not in stopwords.words('english')] for tokens in X]
        X = [tokens + ([0] * (max_words - len(tokens))) if len(tokens) < max_words else tokens[:max_words] for tokens in X]

        return torch.tensor(X, dtype=torch.int32).to(device), torch.tensor(Y).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=vectorize_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=vectorize_batch)

    return train_loader, test_loader, len(vocab)


def test(model, loss_fn, test_loader, device):
    with torch.no_grad():
        y_shuffeled, y_preds, losses = [], [], []
        for X, y in tqdm(test_loader):
            X = X.to(device)
            y = y.to(device)

            outs, probas = model(X)
            loss = loss_fn(outs, y)
            losses.append(loss.item())

            y_shuffeled.append(y)
            y_preds.append(probas.argmax(dim=1))
        
        run_correclty = torch.sum(torch.cat(y_preds) == torch.cat(y_shuffeled))
        accuracy = run_correclty.double() / 25000

        return np.mean(losses), accuracy * 100

def train(model, loss_fn, optimizer, train_loader, test_loader, device, epochs=10):
    logs = []
    for i in range(1, epochs+1):
        model.train()

        losses = []
        running_correct = 0
        for X, y in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)

            outs, probas = model(X)
            loss = loss_fn(outs, y)
            losses.append(loss.item())

            y_pred = probas.argmax(dim=1)
            running_correct += torch.sum(y_pred == y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_accuracy = running_correct.double() / 25000
        print(f'Epoch {i}\n', 85*'-')
        print(f'Train loss: {np.mean(losses)}, accuracy: {epoch_accuracy*100}%')

        model.eval()
        test_loss, test_accuracy = test(model, loss_fn, test_loader, device)
        print(f'Test loss: {test_loss}, accuracy: {test_accuracy}%')

        logs.append({
            'train_loss': np.mean(losses),
            'test_loss': test_loss,
            'train_accuracy': epoch_accuracy.item() * 100,
            'test_accuracy': test_accuracy.item()
        })
    return logs


def acc_loss_curves(train_accs, test_accs, train_losses, test_losses, names=('AccuracyCurve', 'LossCurve')):
    fig1 = plt.figure()
    plt.plot(np.array(test_accs), label="Validation accuracies")
    plt.plot(np.array(train_accs), label="Train accuracies")
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
    device = torch.device('cuda')
    print(f'Device is {device}')

    train_loader, test_loader, length = get_imdb_dataset(device)
    for i, (X, y) in enumerate(train_loader, start=1):
        print(i, X.shape, y.shape)
        break
    
    lstm_classifier = LSTMClassifier(length, 50, 75, 1).to(device)

    epochs = 10
    learning_rate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_classifier.parameters(), lr=learning_rate)

    model_performance = train(lstm_classifier, loss_fn, optimizer, train_loader, test_loader, device, epochs)
    acc_loss_curves(
        train_accs=[log['train_accuracy'] for log in model_performance],
        test_accs=[log['test_accuracy'] for log in model_performance],
        train_losses=[log['train_loss'] for log in model_performance],
        test_losses=[log['test_loss'] for log in model_performance]
    )


if __name__ == '__main__':
    main()