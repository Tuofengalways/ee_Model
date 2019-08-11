import torch
from torchtext import data

SEED = 1370

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)
# LABEL = data.LabelField(dtype=float)
# TypeError: tensor(): argument 'dtype' must be torch.dtype, not type

from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

import random

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

MAX_VOCAB = 25000
TEXT.build_vocab(train_data, max_size=MAX_VOCAB)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                           batch_size=BATCH_SIZE, device=device)

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        # 用断言保证 取的是最后一个 hidden
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 500
HIDDEN_DIM = 512
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# https://stackoverflow.com/questions/50954479/using-cuda-with-pytorch
# 把一个模型或者tensor放到 device里
model = model.to(device)
criterion = model.to(device)


def binary_accurancy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = sum(correct) / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accurancy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_acc += acc.item()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_acc = 0
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accurancy(predictions, batch.label)

            epoch_acc += acc.item()
            epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


EPOCHS_NUM = 500

# ???
best_valid_loss = float('inf')

import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


for epoch in range(EPOCHS_NUM):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print('Epoch', epoch + 1, '| Epoch Time : ', epoch_mins, 'm', epoch_secs, 's')
    print('\t Train Loss： |', train_loss, '  Train  Acc :', train_acc * 100, '%')
    print('\t valid Loss： |', valid_loss, '  valid  Acc :', valid_acc * 100, '%')
