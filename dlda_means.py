import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
from model import vgg
USE_GPU = True
dtype = torch.float

class Omniglot(object):
    def __init__(self,path, transform):
        # 预处理过程

        traindir = os.path.join(path, 'train')
        testdir = os.path.join(path, 'test')

        self.trainset = datasets.ImageFolder(traindir, transform)
        self.testset = datasets.ImageFolder(testdir, transform)
        self.train_size = len(self.trainset)
        self.test_size = len(self.testset)
        self.num_classes = len(self.trainset.classes)
        self.class_to_idx = self.trainset.class_to_idx

    def get_loader(self, train_batch,test_batch,):
        trainloader = DataLoader(self.trainset, batch_size=train_batch, shuffle=True)
        testloader = DataLoader(self.testset, batch_size=test_batch, shuffle=True)
        return trainloader, testloader

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

batch_size2 = 16
batch_size1 = 16

data_path = r'data3'
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
dataset = Omniglot(path=data_path, transform=transform)
trainloader, testloader = dataset.get_loader(batch_size1,batch_size2)
num_classes = dataset.num_classes #种类数量
trainsize = dataset.train_size
testsize = dataset.test_size

'''
class DeepLDA(nn.Module):
    def __init__(self):
        super(DeepLDA, self).__init__()
        self.layer = nn.Sequential(
            torch.nn.Linear(768, 520),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(520, 320),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(320, 256),
        )
    def forward(self, x):
        x = x.view(-1,768)
        H = self.layer(x)
        return H
'''

def comupte(x,y):
    N,f = x.shape
    labels, counts = torch.unique(y, return_counts=True)
    sw = torch.zeros((f,f),dtype=dtype, device=device, requires_grad=True)
    u = torch.mean(x, 0)
    x_bar = x - u
    mean = torch.zeros((num_classes, f), dtype=dtype, device=device, requires_grad=False)
    if N!=1:
        St = x_bar.t().matmul(x_bar) / (N - 1)
    else:
        St = x_bar.t().matmul(x_bar)
    for c,NC in zip(labels,counts):
        xc = x[y==c]
        mean[int(c),:] = torch.mean(xc,0)
        xc_bar =xc - mean[int(c),:]
        sw = sw + xc_bar.t().matmul(xc_bar)
    #sw = sw/num_classes
    sb = St-sw
    temp = sw.pinverse().matmul(sb)
    evals, evecs = torch.eig(temp, eigenvectors=True)
    noncomplex_idx = evals[:, 1] == 0
    evals = evals[:, 0][noncomplex_idx]
    evecs = evecs[:, noncomplex_idx]
    evals, inc_idx = torch.sort(evals)
    evecs = evecs[:, inc_idx]
    return evals,evecs,mean


def Loss(evals,margin):
    n = num_classes-1 #取倒数n-1个最大值
    evals = evals[-n:]
    if margin is not None:
        threshold = torch.min(evals)+margin
        n_eig = torch.sum(evals<threshold)
    loss = torch.mean(evals[:n_eig])
    loss = Variable(loss,requires_grad=True)
    return loss

def predict(x,xc,evecs):
    n,_ = x.shape
    results = []
    for i in range(n):
        coef = xc.matmul(evecs).matmul(evecs.T)
        intercept = -0.5 * torch.diagonal(xc.matmul(coef.t()))
        logit = x[i,:].T.matmul(coef.T)+intercept
        pre = torch.nn.functional.log_softmax(logit,dim=0)
        results.append(torch.argmax(pre).item())
    return results


def distance(x,y):
    init_d = torch.norm(y[0,:]-x)
    k = 0
    for i in range(num_classes):
        d = torch.norm(y[i,:]-x)
        if d < init_d:
            k = i
            init_d=d
    return k

def predict2(x,xc,evecs):
    n, _ = x.shape
    results = []
    xc = xc.matmul(evecs)
    x = x.matmul(evecs)
    for i in range(n):
        label = distance(x[i,:],xc)
        results.append(label)
    return results


if __name__ == '__main__':
    dlda = vgg("vgg16",num_classes=10,init_weights=True)
    dlda = dlda.to(device=device)
    lamb =0.001
    optimizer = optim.SGD(dlda.parameters(),lr=0.01)
    Xbar = torch.zeros((num_classes, 1), device=device, dtype=torch.long)
    margin = 0.001
    epoch =5
    sum =0
    xx=0
    sample_mean =torch.zeros((epoch, 1), device=device)
    for i in range(epoch):
        correct = 0
        for indx, (data, target) in enumerate(trainloader):
            dlda.train()
            x = data.to(device=device, dtype=dtype)
            y = target.to(device=device, dtype=dtype)
            x = dlda(x)  #16 11
            print(x)
            evals,evecs,mean = comupte(x,y)
            loss = Loss(evals,margin)
            loss.backward()
            optimizer.step()
            pred = predict2(x,mean,evecs)

            for j in range(len(y)):
                if pred[j]== y[j]:
                    correct = correct+1
        sample_mean[i,:] =correct / trainsize
        print('accuracy: {:.2%}'.format(correct / trainsize))
    for i in range(epoch):
        xx += float(sample_mean[i,:])
    print('accuracy: {:.2%}'.format(xx/5))