import torch
import time 
import numpy as np 

def compute_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()

def train_model(model, num_epochs, train_iterator, valid_iterator, optimizer, criterion, device):
    train_loss_list, valid_loss_list = [],[]
    valid_loss_min = np.Inf 
    for epoch in range(1, num_epochs+1):

        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in train_iterator:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            train_loss_list.append(train_loss.item())
            
        model.eval()
        for data, target in valid_iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
            valid_loss_list.append(valid_loss.item())
        
        train_loss = train_loss/len(train_iterator.sampler)
        valid_loss = valid_loss/len(valid_iterator.sampler)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format( valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

    return train_loss_list, valid_loss_list