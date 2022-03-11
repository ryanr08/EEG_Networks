import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def train(model, optimizer, train_loader, epoch):
    
    # put the model into training mode
    model.train()
    running_loss = 0.0
    loss = None
    for i, data in enumerate(train_loader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()
    
        # reshape inputs for time series convolution
        inputs = torch.transpose(inputs, 1, 3)

        # forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        # backward + optimize
        loss.backward() # backward to get gradient values
        optimizer.step() # does the update

        # accumulate loss
        running_loss += loss.item()
        
        # verbose
        if i % 10 == 0:
            print('Training Progress: \tEpoch {} [{}/{} ({:.2f}%)]\t\tLoss: {:.5f}'.format(
                epoch+1, i*len(inputs), len(train_loader.dataset), 100.*i/len(train_loader), loss.data))

    return model

def evaluate(model, data_loader, mode):
    
    # put the model into evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = torch.transpose(inputs, 1, 3)
            labels = labels.type(torch.float)
            
            # calculate outputs by running images through the network
            outputs = model(inputs)
            
            # calculate validate loss
            test_loss += F.cross_entropy(outputs, labels).data
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, dim=1)
            _, label_indeces = torch.max(labels.data, dim=1)
            total += labels.size(0)
            correct += (predicted == label_indeces).sum().item()
        
        # average test_loss
        test_loss /= total

    if mode == 'train':
        print('\tTrain loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(data_loader.dataset), 100 * correct // total))
    elif mode == 'val':
        print('\tValidation loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(data_loader.dataset), 100 * correct // total))
    
    elif mode == 'test':
        print('\tTest loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), 100 * correct // total))
    
    else:
        pass
    
    return [test_loss, 1.*correct/total]   

def train_and_evaluate(model, optimizer, data_loaders, num_epochs=10):

    # unpackage data loaders
    train_loader = data_loaders[0]
    val_loader = data_loaders[1]
    test_loader = data_loaders[2]
    
    # initialize book keeping dictionary
    metrics = {}
    metrics['train'] = []
    metrics['val']  = []
    metrics['test'] = []

    # evaluate for each epoch and record
    for epoch in range(num_epochs):
        model = train(model, optimizer, train_loader, epoch)

        metrics['train'].append(evaluate(model, train_loader, mode='train'))
        metrics['val'].append(evaluate(model, val_loader, mode='val'))
        metrics['test'].append(evaluate(model, test_loader, mode='test'))

    metrics['train'] = np.array(metrics['train'])
    metrics['val']  = np.array(metrics['val'])
    metrics['test']  = np.array(metrics['test'])

    print('Best validation accuracy:')
    print(np.amax(metrics['val'][:, 1].data))

    print('Best test accuracy:')
    print(np.amax(metrics['test'][:, 1].data))

    plot(metrics, num_epochs)


def plot(metrics, num_epochs):

    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
    
    ax[0].plot(range(num_epochs), metrics['train'][:, 0], range(num_epochs), metrics['val'][:, 0])
    ax[0].legend(['Train','Validation'])
    ax[0].set_title('Loss')

    ax[1].plot(range(num_epochs), metrics['train'][:, 1], range(num_epochs), metrics['val'][:, 1])
    ax[1].legend(['Train','Validation'])
    ax[1].set_title('Accuracy')