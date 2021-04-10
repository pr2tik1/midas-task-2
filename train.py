import torch
import time 
import numpy as np 

def compute_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train_model(model, num_epochs, train_iterator, valid_iterator, optimizer, criterion, device, model_name):
    train_loss_list, valid_loss_list = [],[]
    train_acc_list, valid_acc_list = [],[]
    
    valid_loss_min = np.Inf 
    for epoch in range(1, num_epochs+1):

        train_loss, valid_loss = 0.0, 0.0
        train_acc, valid_acc = 0.0 , 0.0

        model.train()
        for data, target in train_iterator:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            acc = compute_accuracy(output, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc.item()

        model.eval()
        for data, target in valid_iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_acc = compute_accuracy(output, target)
            
            valid_loss += loss.item()
            valid_acc += val_acc.item()
        
        train_loss = train_loss/len(train_iterator.sampler)
        valid_loss = valid_loss/len(valid_iterator.sampler)
        valid_loss_list.append(valid_loss)
        train_loss_list.append(train_loss)


        train_acc = train_acc/len(train_iterator.sampler)
        valid_acc = valid_acc/len(valid_iterator.sampler)
        valid_acc_list.append(valid_acc)
        train_acc_list.append(train_acc)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format( valid_loss_min, valid_loss))
            torch.save(model.state_dict(), model_name)
            valid_loss_min = valid_loss
        
    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list
