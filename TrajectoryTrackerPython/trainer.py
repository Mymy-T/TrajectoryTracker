import torch
import torch.nn as nn

def train_epoch(model, train_loader, lr, epoch, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    
    total_loss = 0
    for idx, batch in enumerate(train_loader): # batch is (inputs, outputs)
        optimizer.zero_grad()
        pred = model(batch[0])
        loss = loss_fn(pred, batch[1])
        loss.backward()
        total_loss += loss.item()
        if verbose:
            print(f"Avg loss for batch {idx + 1}: {loss.item():.4f}")
    avg_loss = total_loss/len(train_loader)
    if verbose:
        print(f"Avg loss for epoch {epoch}: {avg_loss:.4f}") # epoch is indexed from 1
        print("------------------------------------------------")
    return avg_loss
    

def evaluate(model, data_loader, valid, verbose=True):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.inference_mode():
        total_loss = 0
        for idx, batch in enumerate(data_loader): # batch is (inputs, outputs)
            pred = model(batch[0])
            loss = loss_fn(pred, batch[1])
            total_loss += loss.item()
            if verbose:
                print(f"Avg loss for batch {idx + 1}: {loss.item():.4f}")
        avg_loss = total_loss/len(data_loader)
        if verbose:
            if valid:
                name = "validation"
            else:
                name = "test"
            print(f"Avg {name} loss: {avg_loss:.4f}")
        
        return avg_loss
