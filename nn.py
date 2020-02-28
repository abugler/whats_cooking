import numpy
from collections import OrderedDict
import torch.nn as nn
from torch.nn.modules.loss import NLLLoss
from torch.optim import SGD
import torch
from config import n_layers, n_epochs, n_hidden, lr, momentum, batch_size
import time

# Note: The neural network MUST be trained on gpu

if n_layers != len(n_hidden):
    raise ValueError("List of number of hidden units does not match the number of indicated layers")

class CuisineClassifier(nn.Module):
    def __init__(self, n_input, n_output):
        super(CuisineClassifier, self).__init__()
        self.n_layers = n_layers
        self.n_epochs = n_epochs

        self.input = nn.Sequential(
            nn.Linear(n_input, n_hidden[0]),
            nn.Sigmoid()
        ).cuda()
        hidden_dict = OrderedDict()
        for i in range(n_layers - 1):
            hidden_dict[f"linear{i}"] = nn.Linear(n_hidden[i], n_hidden[i + 1])
            hidden_dict[f"softmax{i}"] = nn.Sigmoid()
        self.hidden = nn.Sequential(
            hidden_dict
        ).cuda()
        self.output = nn.Sequential(
            nn.Linear(n_hidden[-1], n_output),
            nn.LogSoftmax(dim=1)
        ).cuda()

    def forward(self, X):
        return self.output(self.hidden(self.input(X)))

output_template = "#################################\n" + \
                  "#                               #\n" + \
                  "#          Epoch: {4:3}           #\n" + \
                  "#     Time Elapsed:  {0:2}:{1:2}      #\n" + \
                  "#      Total Time:  {5:2}:{6:2}       #\n" + \
                  "#   Validation Loss:  {2:7.5f}   #\n" + \
                  "#  Min Training Loss:  {3:7.5f}  #\n" + \
                  "#                               #\n" + \
                  "#################################"

def train(model, training_features, training_targets,
          validation_features, validation_targets):
    # loss is negative log likelihood
    loss = NLLLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    # cast to torch tensors
    training_features = torch.from_numpy(training_features).float().cuda()
    training_targets = torch.from_numpy(training_targets).long().cuda()
    validation_features = torch.from_numpy(validation_features).float().cuda()
    validation_targets = torch.from_numpy(validation_targets).long().cuda()

    min_valid_loss = 1e10
    train_start = int(time.time())
    for epoch in range(model.n_epochs):
        epoch_start = int(time.time())
        print(f"Starting Epoch {epoch}")
        # Do mini-batch gradient descent
        perm = torch.randperm(training_features.size()[0])
        min_training_loss = 1e10
        for i in range(0, training_features.size()[0], batch_size):
            optimizer.zero_grad()
            # Find batches
            batch_indices = perm[i:max(i + batch_size, training_features.size()[0])]
            batch_features = training_features[batch_indices, :]
            batch_targets = training_targets[batch_indices]
            # forward pass
            predicted = model(batch_features)
            # predict loss
            batch_loss = loss(predicted, batch_targets)
            # backprop
            batch_loss.backward()
            optimizer.step()
            if batch_loss.item() < min_training_loss:
                min_training_loss = batch_loss.item()

        # validation loss
        valid_loss = loss(model(validation_features), validation_targets).item()
        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            torch.save(model, "best_nn")
        epoch_end = int(time.time())
        epoch_elapsed = epoch_end - epoch_start
        train_elapsed = epoch_end - train_start
        print(output_template.format(
            int(epoch_elapsed / 60), epoch_elapsed % 60, valid_loss, min_training_loss, epoch,
            int(train_elapsed / 60), train_elapsed % 60
        ))


