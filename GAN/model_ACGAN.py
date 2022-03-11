import os, time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from preprocessing import data_prep


class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.linear = nn.Linear(104, 1200)
        self.bn = nn.BatchNorm1d(1200)
        self.dropout = nn.Dropout2d(p=0.5)
        self.transconv1 = nn.ConvTranspose2d(400, 200, (6, 1), stride=(2, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(200)
        self.transconv2 = nn.ConvTranspose2d(200, 100, (10, 1), stride=(2, 1), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(100)
        self.transconvp = nn.ConvTranspose2d(100, 50, (5, 1), stride=(1, 1), padding=(2, 0))
        self.bnp = nn.BatchNorm2d(50)
        self.transconv3 = nn.ConvTranspose2d(50, 25, (9, 1), stride=(3, 1), padding=(3, 0))
        self.bn3 = nn.BatchNorm2d(25)
        self.transconv4 = nn.ConvTranspose2d(25, 22, (3, 1), stride=(3, 1), padding=(1, 0))

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward
    def forward(self, input, label):
        x = torch.cat([input, label],1)
        x = self.linear(x)
        x = self.bn(x)
        x = torch.reshape(x,(-1,400,3,1))
        x = self.dropout(self.bn1(F.leaky_relu(self.transconv1(x))))
        x = self.dropout(self.bn2(F.leaky_relu(self.transconv2(x))))
        x = self.dropout(self.bnp(F.leaky_relu(self.transconvp(x))))
        x = self.dropout(self.bn3(F.leaky_relu(self.transconv3(x))))
        x = torch.tanh(self.transconv4(x))
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(22,25,(11,1),padding=(5,0),stride=(3,1))
        self.bn1 = nn.BatchNorm2d(25)
        self.dropout = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(25, 50, (11, 1), padding=(4, 0),stride=(3,1))
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 100, (3, 1), padding=(1, 0),stride=(3,1))
        self.bn3 = nn.BatchNorm2d(100)
        self.conv4 = nn.Conv2d(100, 200, (4, 1), padding=(0, 0), stride=(3, 1))
        self.bn4 = nn.BatchNorm2d(200)
        self.flatten = nn.Flatten()



        self.dense = nn.Linear(600,1)
        self.aux1 = nn.Linear(600,4)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.dropout((F.relu(self.conv1(input)))) #(25,83,1)
        x = self.dropout((F.relu(self.conv2(x)))) #(50,27,1)
        x = self.dropout((F.relu(self.conv3(x))))#(100,9,1)
        x = self.dropout((F.relu(self.conv4(x))))#(200,3,1)
        x = self.flatten(x) #(600,)

        out = self.dense(x) #(1,)
        auxout = self.aux1(x)


        return out, auxout

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



def show_result(i, show = True):
    G.eval()
    test_images = G(test_z_, test_y_label_)
    G.train()

    ch8 = test_images[:,17,:].cpu().detach().numpy()
    avg = np.mean(ch8,axis=0)
    plt.plot(np.arange(250),avg)
    plt.ylim([-0.1, 0.2])
    if show & (i % 10 == 0):
        plt.show()
    else:
        plt.close()

def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

#curve
def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # training parameters
    batch_size = 64
    lr_G = 0.0001
    lr_D = 0.00008
    train_epoch = 300
    gp = 10 #gradient penalty
    n_critic = 5 #number of iterations of the critic per generator iteration

    # test: fixed noise & label
    test_z_ = torch.randn(10, 100)
    test_y_label_ = torch.zeros(10, 4)
    test_y_label_[:, 1] = 1  # label
    test_z_, test_y_label_ = test_z_.cuda(), test_y_label_.cuda()

    # data_loader
    X_test = np.load(r"true_data/X_test.npy")
    y_test = np.load(r"true_data/y_test.npy")
    person_train_valid = np.load(r"true_data/person_train_valid.npy")
    X_train_valid = np.load(r"true_data/X_train_valid.npy")
    y_train_valid = np.load(r"true_data/y_train_valid.npy")
    person_test = np.load(r"true_data/person_test.npy")

    y_train_valid -= 769
    y_test -= 769

    X_train_valid_prep,y_train_valid_prep = data_prep(X_train_valid,y_train_valid,2,2,True)
    X_test_prep,y_test_prep = data_prep(X_test,y_test,2,2,True)

    ind_valid = np.random.choice(8460, 1500, replace=False)
    ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))
    (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid]
    (y_train_compact, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)

    x_train = torch.from_numpy(x_train)
    x_valid = torch.from_numpy(x_valid)
    x_test = torch.from_numpy(x_test)
    y_train_compact = torch.from_numpy(y_train_compact)
    y_valid = torch.from_numpy(y_valid)
    y_test = torch.from_numpy(y_test_prep)


    traindataset = torch.utils.data.TensorDataset(x_train,y_train_compact)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=64, shuffle=True, num_workers=4)

    # network
    G = generator()
    D = discriminator()
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)

    G.cuda()
    D.cuda()

    Gpath = r'cACGAN_results/g.pkl'# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Dpath = r'cACGAN_results/d.pkl'

    G.load_state_dict((torch.load(Gpath)))
    D.load_state_dict((torch.load(Dpath)))


    # Binary Cross Entropy loss
    aux_loss = nn.CrossEntropyLoss()
    aux_loss.cuda()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # results save folder
    root = 'cACGAN_results/'
    model = 'cACGAN_'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'Fixed_results'):
        os.mkdir(root + 'Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []


    print('training start!')
    i = 0
    start_time = time.time()
    acc_list = []
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []

        D_real_dis_loss = []
        D_real_aux_loss = []
        G_dis_loss = []
        G_aux_loss = []


        epoch_start_time = time.time()
        y_real_ = torch.ones(batch_size)
        y_fake_ = torch.zeros(batch_size)
        y_real_ -= torch.Tensor(0.01*np.random.random(y_real_.shape))
        y_fake_ += torch.Tensor(0.01 * np.random.random(y_fake_.shape))
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        for x_, y_compact in train_loader:
            y_compact = torch.as_tensor(y_compact,dtype=torch.long)
            y_ = F.one_hot(y_compact.to(torch.int64), num_classes=4)

            # train discriminator D\
            i += 1
            D.zero_grad()

            mini_batch = x_.size()[0]

            if mini_batch != batch_size:
                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)
                y_real_ -= torch.Tensor(0.01 * np.random.random(y_real_.shape))
                y_fake_ += torch.Tensor(0.01 * np.random.random(y_fake_.shape))
                y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            x_ = x_.type(torch.FloatTensor)
            y_ = y_.type(torch.FloatTensor)
            x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
            y_compact = Variable(y_compact.cuda())

            #D_result = D(x_, y_).squeeze()
            D_result,aux_result = D(x_)
            # D_real_loss = BCE_loss(D_result, y_real_)
            D_real_loss = -torch.mean(D_result)
            D_aux_loss1 = aux_loss(aux_result, y_compact)

            accuracy = compute_acc(aux_result, y_compact)

            z_ = torch.randn((mini_batch, 100)).view(-1, 100)
            z_ = Variable(z_.cuda())

            G_result = G(z_, y_)
            D_result, aux_result2 = D(G_result)


            D_fake_loss = torch.mean(D_result)
            D_aux_loss2 = aux_loss(aux_result2, y_compact)




            #gradient penalty
            alpha = torch.rand((mini_batch, 1,1,1)).cuda()

            x_hat = alpha * x_ + (1-alpha) * G_result
            # x_hat.requires_grad = True

            pre_hat,_ = D(x_hat)

            gradients = grad(outputs=pre_hat, inputs=x_hat, grad_outputs=torch.ones(pre_hat.size()).cuda(),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = gp * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            D_train_loss = D_real_loss + D_fake_loss + gradient_penalty + 1 *(D_aux_loss1 + D_aux_loss2)#?????????????????????#?????????????????????#?????????????????????#?????????????????????#?????????????????????

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.item())
            D_real_dis_loss.append((D_real_loss + D_fake_loss).item())
            D_real_aux_loss.append(D_aux_loss1.item())

            # train generator G

            if (i % n_critic) == 0:

                G.zero_grad()

                z_ = torch.randn((mini_batch, 100)).view(-1, 100)
                y_ = (torch.rand(mini_batch, 1) * 4).type(torch.LongTensor).squeeze()
                y_ = Variable(y_.cuda())

                y_label_ = F.one_hot(y_.to(torch.int64), num_classes=4)

                z_ = z_.type(torch.FloatTensor)
                y_label_ = y_label_.type(torch.FloatTensor)
                z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

                G_result = G(z_, y_label_)
                D_result,aux_result3 = D(G_result)

                # G_train_loss = BCE_loss(D_result, y_real_)
                G_dis = -torch.mean(D_result)
                G_aux =  aux_loss(aux_result3,y_)
                G_train_loss = G_dis +  1*G_aux #?????????????????????#?????????????????????#?????????????????????#?????????????????????#?????????????????????#?????????????????????#?????????????????????

                G_train_loss.backward()
                G_optimizer.step()

                G_losses.append(G_train_loss.item())
                G_dis_loss.append(G_dis.item())
                G_aux_loss.append(G_aux.item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        acc_list.append(accuracy)


        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f, accuracy: %.3f,D_dis_loss: %.3f, D_aux_loss: %.3f, G_dis_loss: %.3f,G_aux_loss: %.3f' % (
        (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses)),accuracy,torch.mean(torch.FloatTensor(D_real_dis_loss)),torch.mean(torch.FloatTensor(D_real_aux_loss)),torch.mean(torch.FloatTensor(G_dis_loss)),torch.mean(torch.FloatTensor(G_aux_loss))))
        show_result((epoch+1),show = True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    np.save(r'cACGAN_results/accuracy', acc_list)
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), root + model + 'generator_param.pkl')
    torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
    with open(root + model + 'train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')