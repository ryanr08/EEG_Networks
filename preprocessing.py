import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from torch.autograd import Variable


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
#         self.X = torch.cuda.FloatTensor(X)
#         self.Y = torch.cuda.LongTensor(Y)
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
        return x, y

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def load_data(data_path, subjects=[1,2,3,4,5,6,7,8,9], verbose=False):

    X_train_valid = np.load(data_path + "X_train_valid.npy")
    y_train_valid = np.load(data_path + "y_train_valid.npy") - 769
    person_train_valid = np.load(data_path + "person_train_valid.npy")

    X_test = np.load(data_path + "X_test.npy")
    y_test = np.load(data_path + "y_test.npy") - 769
    person_test = np.load(data_path + "person_test.npy")

    X_train_valid_subjects = np.empty(shape=[0, X_train_valid.shape[1], X_train_valid.shape[2]])
    y_train_valid_subjects = np.empty(shape=[0])
    X_test_subjects = np.empty(shape=[0, X_test.shape[1], X_test.shape[2]])
    y_test_subjects = np.empty(shape=[0])

    for s in subjects:

        # extract subject data
        X_train_valid_subject = X_train_valid[np.where(person_train_valid == s-1)[0], :, :]
        y_train_valid_subject = y_train_valid[np.where(person_train_valid == s-1)[0]]
        X_test_subject = X_test[np.where(person_test == s-1)[0], :, :]
        y_test_subject = y_test[np.where(person_test == s-1)[0]]

        # stack
        X_train_valid_subjects = np.concatenate((X_train_valid_subjects, X_train_valid_subject), axis=0)
        y_train_valid_subjects = np.concatenate((y_train_valid_subjects, y_train_valid_subject))
        X_test_subjects = np.concatenate((X_test_subjects, X_test_subject), axis=0)
        y_test_subjects = np.concatenate((y_test_subjects, y_test_subject))

    if verbose:
        print ('Training/Valid data shape: {}'.format(X_train_valid_subjects.shape))
        print ('Test data shape: {}'.format(X_test_subjects.shape))

    return X_train_valid_subjects, y_train_valid_subjects, X_test_subjects, y_test_subjects
    
def data_prep(X,y,sub_sample,average,noise):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:500]
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)    
    total_X = X_max
    total_y = y
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] +                             (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    return total_X,total_y

def random_split_train_test(X_train_valid, y_train_valid):
    ## Random splitting and reshaping the data
    num_sample = X_train_valid.shape[0]
    num_valid = round(num_sample*0.2)
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(num_sample, num_valid, replace=False)
    ind_train = np.array(list(set(range(num_sample)).difference(set(ind_valid))))

    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
    
    return x_train, x_valid, y_train, y_valid

def one_hot_label(y_train, y_valid, y_test):
    y_train = torch.tensor(y_train).to(torch.int64)
    y_valid = torch.tensor(y_valid).to(torch.int64)
    y_test = torch.tensor(y_test).to(torch.int64)

    y_train = F.one_hot(y_train, 4)
    y_valid = F.one_hot(y_valid, 4)
    y_test = F.one_hot(y_test, 4)
    
    return y_train, y_valid, y_test

def reshape_data(x_train, x_valid, x_test):
    # Adding width of the segment to be 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Reshaping the training and validation dataset
    x_train = np.swapaxes(x_train, 1,3)
    x_train = np.swapaxes(x_train, 1,2)
    x_valid = np.swapaxes(x_valid, 1,3)
    x_valid = np.swapaxes(x_valid, 1,2)
    x_test = np.swapaxes(x_test, 1,3)
    x_test = np.swapaxes(x_test, 1,2)
    
    return x_train, x_valid, x_test

def main_prep(X_train_valid, y_train_valid, X_test, y_test, sub_sample, average, noise):    
    ## Preprocessing the dataset
    X_train_valid_prep,y_train_valid_prep = data_prep(X_train_valid,y_train_valid,2,2,True)
    X_test_prep,y_test_prep = data_prep(X_test,y_test,2,2,True)
    
    x_train, x_valid, y_train, y_valid = random_split_train_test(X_train_valid_prep, y_train_valid_prep)
    y_train, y_valid, y_test = one_hot_label(y_train, y_valid, y_test_prep)
    x_train, x_valid, x_test = reshape_data(x_train, x_valid, X_test_prep)
    print('Shape of x_train:',x_train.shape)
    print('Shape of x_valid:',x_valid.shape)
    print('Shape of x_test:',x_test.shape)
    
    print('Shape of y_train:',y_train.shape)
    print('Shape of y_valid:',y_valid.shape)
    print('Shape of y_test:',y_test.shape)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def dataloader_setup(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=32):
    # transformations
    transformations = transforms.Compose([
                        transforms.RandomErasing(p=.99,
                                                 scale=(.02,.08),
                                                 ratio=(.025,.026),
                                                 value=0),
                        AddGaussianNoise(mean=0., std=1.),
                      ])

    # load training dataset
    train_dataset = Dataset(X_train, y_train, transform=transformations)
    train_loader = DataLoader(train_dataset, batch_size)

    # load validation dataset
    val_dataset = Dataset(X_valid, y_valid)
    val_loader = DataLoader(val_dataset, X_valid.shape[0])

    # load test dataset
    test_dataset = Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, len(test_dataset))

    # package up
    data_loaders = [train_loader, val_loader, test_loader]

    return data_loaders