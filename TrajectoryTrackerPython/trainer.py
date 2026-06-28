import torch
import torch.nn as nn

def train(model, train_loader, lr, epoch):

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        model.train()
        output = model(batch[0])


