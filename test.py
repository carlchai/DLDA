import os
import random
import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn import svm

import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
from sklearn.metrics import accuracy_score
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import torchvision.datasets as datasets


USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)
transform = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor(),  # 把数据转换成张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_dataset = datasets.ImageFolder(root='./cifar10_new/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./cifar10_new/val', transform=transform)
test_dataset = datasets.ImageFolder(root='./cifar10_new/test', transform=transform)
template_dataset = datasets.ImageFolder(root='./cifar10_new/template', transform=transform)
num_class = len(train_dataset.classes)
outfeatures = 128
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=True)
templateloader = torch.utils.data.DataLoader(template_dataset, batch_size=num_class, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        out = self.relu(x + residual)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=outfeatures):
        super(ResNet18, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

        )
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resblock(in_channels=64),
            resblock(in_channels=64),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            resblock(in_channels=128),
            resblock(in_channels=128),
            nn.Dropout(0.25)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),
            resblock(in_channels=256),
            resblock(in_channels=256),

        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=(2, 2), padding=1),
            resblock(in_channels=512),
            resblock(in_channels=512),
            nn.Dropout(0.5)
        )
        self.block6 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=outputs),
        )
        self.initial(self.block1)
        self.initial(self.fc)

    def initial(self, *models):
        for m in models:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.reshape(x.shape[0], -1)
        features = self.fc(x)
        return features

def get_template():
    template = torch.zeros((num_class,outfeatures),device=device)
    with torch.no_grad():
        for data,label in templateloader:
            data = data.to(device=device)
            label = label.to(device=device)
            output = net(data)
            for i in range(len(label)):
                template[int(label[i]),:] = output[i,:]
    return template


def lda_loss2(gamma, x, y, xcmean):
    n, d = x.shape
    dintra = 0
    dinter = 0
    for i in range(n):
        dintra += torch.norm(x[i, :] - xcmean[int(y[i]), :], p=2)
    for i in range(num_class):
        xc = Hard_negative_mining(xcmean[i, :],i, xcmean)
        dinter += torch.norm(xcmean[i, :] - xc, p=2)
    loss = gamma * dintra / dinter
    return loss


def train(epoch):
    best_loss = 1000000
    record_loss = []
    record_acc = []
    for i in range(epoch):
        net.train()
        acc = 0
        total_loss = 0
        tbar = tqdm.tqdm(trainloader)
        for batch_id,(inputs,target) in enumerate(tbar):
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            features = net(inputs)
            xc,evals,evecs = T_LDA(features,target,num_class,a,lamb,template)
            loss = lda_loss2(gamma,features,target,template)
            result = predict(features,xc,evecs)
            correct = result.eq(target).sum().item()
            nums = len(target)
            results = correct / nums
            acc+=results
            loss.backward()
            optimizer.step()
            total_acc = acc/(batch_id+1)
            total_loss += loss.item()
            tbar.set_postfix(epoch=i,total_loss="{:.3f}".format(total_loss),total_acc="{:.4f}".format(np.array(total_acc)))
        if best_loss>total_loss:
            best_loss=total_loss
            checkpoint = {'epoch': i, 'best_loss': best_loss, 'state_dict': net.state_dict()}
            print("find best loss")
        record_loss.append(total_loss)
        record_acc.append(total_acc)
        torch.save(checkpoint,model_path)
    return record_acc,record_loss


def _testonI2CS():
    acc=0
    total_loss=0
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    for batch_id,(inputs,target) in enumerate(testloader):
        inputs = inputs.to(device)
        target = target.to(device)
        features = net(inputs)
        xc, evals, evecs = T_LDA(features, target, num_class, a, lamb, template)
        result = predict(features, xc, evecs)
        correct = result.eq(target).sum().item()
        nums = len(target)
        results = correct / nums
        acc += results
        total_acc = acc / (batch_id + 1)
    print('acc: %.2f%%' % (100. * total_acc))

def get_projection_matrix():
    tt_x = []
    tt_y = []
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            feas = net(inputs)
            tt_x.append(feas)
            tt_y.append(targets)
    total_x = tt_x[0]
    total_y = tt_y[0]
    for i in range(len(tt_x) - 1):
        total_x = torch.cat((total_x, tt_x[i + 1]), 0)  # total_numxfeatures
        total_y = torch.cat((total_y, tt_y[i + 1]), 0)
    Xc_mean, evals, evecs = T_LDA(total_x,total_y,num_class,a,lamb,template)
    return Xc_mean, evecs

def compute_accuracy(xc_mean,evecs):
    tt_x = []
    tt_y = []
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            feas = net(inputs)
            tt_x.append(feas)
            tt_y.append(targets)
    total_x = tt_x[0]
    total_y = tt_y[0]
    for i in range(len(tt_x) - 1):
        total_x = torch.cat((total_x, tt_x[i + 1]), 0)  # total_numxfeatures
        total_y = torch.cat((total_y, tt_y[i + 1]), 0)
    res = predict(total_x,xc_mean,evecs)
    correct = res.eq(total_y).sum().item()
    nums = len(total_y)
    test_acc = correct / nums
    print('acc: %.2f%%' % (100. * test_acc))
    return total_x, total_y




def T_LDA(X,y,n_classes,lamb,a,template):
    X = X.view(X.shape[0], -1)
    N, D = X.shape
    labels, counts = torch.unique(y, return_counts=True)
    Xc_m = torch.zeros((n_classes, D), dtype=X.dtype, device=X.device, requires_grad=False)
    Sw = torch.zeros((D, D), dtype=X.dtype, device=X.device, requires_grad=True)
    Sb = torch.zeros((D, D), dtype=X.dtype, device=X.device, requires_grad=True)
    for c, Nc in zip(labels, counts):
        Xc = X[y == c]
        temp = template[int(c), :]
        mean = torch.mean(Xc, 0)
        Xc_m[int(c), :] = (1 - a) * mean + a * temp
        Xc_bar = Xc - Xc_m[int(c), :]
        Sw = Sw + Xc_bar.t().matmul(Xc_bar)
    for i in range(N):
        hn = Hard_negative_mining(X[i, :], y[i], Xc_m)
        hn = torch.reshape(hn, (outfeatures, 1))
        xc_i = torch.reshape(X[i, :], (outfeatures, 1))
        sb_i = xc_i - hn
        Sb = Sb + sb_i.matmul(sb_i.t())
    Sw += torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False) * lamb
    temp = Sw.pinverse().matmul(Sb)
    evals, evecs = torch.linalg.eig(temp)
    evals = evals.real
    evecs = evecs.real
    evals, indx = torch.sort(evals)
    evecs = evecs[:, indx]
    evals = evals[-(num_class - 1):]  # 从小到大
    evecs = evecs[:, -(num_class - 1):]
    return Xc_m, evals, evecs

def lda_loss(evals, n_classes, n_eig=None, margin=None):
    n_components = n_classes - 1
    evals = evals[-n_components:]
    if margin is not None:
        threshold = torch.min(evals) + margin
        n_eig = torch.sum(evals < threshold)
    loss = -torch.mean(evals[:n_eig])  # small eigen values are on left
    return loss

def predict(x,xc,evecs):
    coef = xc.matmul(evecs).matmul(evecs.t())
    intercept = -0.5*torch.diagonal(xc.matmul(coef.t()))
    logit = x.matmul(coef.t())+intercept
    proba = nn.functional.softmax(logit,dim=1)
    results = torch.argmax(proba,dim=1)
    return results

def Hard_negative_mining(x,y, xc):
    n, d = xc.shape
    ls = []
    for i in range(n):
        dis = torch.dist(x,xc[i,:])
        ls.append(dis)
    ls = torch.tensor(ls)
    k = torch.argsort(ls)[0]
    if k!=y:
        hn = xc[k, :]
    else:
        k = torch.argsort(ls)[1]
        hn = xc[k, :]
    return hn

if __name__ == '__main__':
    from resnet50 import Resnet50
    #net = ResNet18(in_channels=3, resblock=ResidualBlock)
    net = Resnet50(outfeatures)
    net = net.to(device)
    model_path = 'stl_res50'  #I2cs 70.4
    gamma = 2
    a = 0.3 #混合投影系数
    lamb = 0.01
    n_eig = 8
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    template = get_template()
    acc,loss =train(1)
    _testonI2CS()
    Xc_mean, evecs = get_projection_matrix()
    xx,yy = compute_accuracy(Xc_mean,evecs)
    #acc_I2cs = [0.43806348872641293, 0.5434846723804378, 0.6105825860696691, 0.6362032361091745, 0.6661677646867448, 0.6975631869887525, 0.7167688253799498, 0.7329570417799545, 0.7505335782092396, 0.762797798004973, 0.7730736265308376, 0.7810320934856617, 0.7968153151288576, 0.8050492402456311, 0.8165322009638177, 0.8217531520874001, 0.8316582082084263, 0.8450467729683724, 0.8490096668612893, 0.8569767574857082, 0.8685145465874234, 0.8678348652601553, 0.8759345788453942, 0.8842123939016314, 0.8889873651439626, 0.8884207446638548, 0.8885580879490843, 0.8948159218720952, 0.8995430998297778, 0.9087642989519428, 0.9128914510567717, 0.9151691437476761, 0.9164478070098997, 0.9184199948584774, 0.9257567406232571, 0.9290237951553496, 0.9308056268445576, 0.9333548743522262, 0.9345364078621724, 0.9347781883249674, 0.9366373153263851, 0.9369057383894543, 0.9375864636346207, 0.9402973368655885, 0.9456091542340583, 0.9458501631053635, 0.9439278563772309, 0.9466977336633203, 0.9487396370724113, 0.9494634352778166, 0.9513395372920151, 0.9527672992627579, 0.9529406804093465, 0.9529153540533788, 0.9569419723258273, 0.9563617809130414, 0.9578772773749769, 0.9599868085087842, 0.9607732871758227, 0.962302354570436, 0.9623404802675917, 0.9639544227261109, 0.9643802958408859, 0.9647448047383342, 0.9648030372019659, 0.9636340307067996, 0.9643591451565114, 0.9654962894619121, 0.9662869438005438, 0.967119627181516, 0.9674303516133342, 0.9671872549062326, 0.9681003653531093, 0.9687873541056656, 0.9686562743278259, 0.9684693222485592, 0.9696973327988474, 0.9677264611945762, 0.9687682912570877, 0.97002920791504, 0.9699889943820877, 0.9713164948468581, 0.9720478728039599, 0.9715648111434514, 0.9722687749087888, 0.9723870553454407, 0.9723374465514035, 0.9728035331991308, 0.9729377447306656, 0.9734638793514129, 0.9740378072568089, 0.9745558628607781, 0.9752924604073713, 0.9754815003224344, 0.975717289604248, 0.9754530422127718, 0.9762922614275422, 0.9756992706735684, 0.9766199608721418, 0.976334562796291, 0.9764580628224344, 0.9769241494701617, 0.9769601873315207, 0.9774231422255532, 0.9779832268718628, 0.9785263365053912, 0.9781195262391941, 0.978802339320157, 0.978043547171291, 0.9781840222102157, 0.9782738445372048, 0.9780921120474297, 0.9789587454539646, 0.978867879209077, 0.9791786036408952, 0.9792916644880555, 0.9790496116988521, 0.9788307974298196, 0.9795600875511248, 0.979696386918456, 0.979696386918456, 0.979761926807376, 0.9797872531633435, 0.9805387378869214, 0.980359093232943, 0.9808251798806702, 0.981088383354248, 0.9808199602911786, 0.9807173386230015, 0.9810734961772635, 0.9812097955445946, 0.9814688233465794, 0.9816273173161832, 0.9812077077087981, 0.9815205199764129, 0.9813134611306702, 0.9814275658957287, 0.9814254780599321, 0.9814508044158998, 0.981679013946017, 0.9816231416445901, 0.9818027862985684, 0.9820056694727181, 0.981603034878114, 0.9821842702087981, 0.9816865936977134, 0.9817382903275469, 0.9824051723136271, 0.982576465624419, 0.9824526932718673, 0.9824009966420338, 0.9824030844778305, 0.9825869048034022, 0.9822550296872095, 0.9821863580445946, 0.9825816852139105, 0.9824263229980015, 0.9826059676519798, 0.9829177360016964, 0.9828712589613544, 0.9829811880548196, 0.9830276650951617, 0.9827412231014129, 0.9828976292352203, 0.9830266211772635, 0.9834302996897658, 0.9831375941886271, 0.9828289575926056, 0.9829780563011248, 0.9832475232820923, 0.9829621252062419, 0.9831830273110708, 0.9833151510068089, 0.9834420551130555, 0.9834895760712957, 0.9835075950019753, 0.9834091490053912, 0.9836078565079476, 0.98351699026306, 0.9836036808363544, 0.9837653065596533, 0.9834261240181725, 0.9836036808363544, 0.9835867058235731, 0.983777061982943, 0.9835856619056749, 0.9839598383906162, 0.983912317432376, 0.9839144052681725, 0.9840708114019798, 0.9840517485534022, 0.9837791498187396, 0.9839483552937349, 0.9838044761747071, 0.9839355559525469, 0.9842018911798196, 0.9843825797516964, 0.9843635169031185, 0.984120420196017, 0.9839799451570923]
    #loss_I2cs_STL10= [6845.461654663086, 5751.503601074219, 5010.1088790893555, 4484.133865356445, 4161.675857543945, 3735.0382690429688, 3461.968048095703, 3241.926284790039, 3034.693801879883, 2841.4309272766113, 2741.0985984802246, 2631.0070610046387, 2450.1287117004395, 2377.169319152832, 2252.6312713623047, 2194.02347946167, 2090.029022216797, 1971.0137329101562, 1915.3868713378906, 1833.602653503418, 1743.0215797424316, 1700.6263999938965, 1657.4883136749268, 1545.8799724578857, 1528.052833557129, 1503.0181293487549, 1516.8738441467285, 1446.169267654419, 1402.7022190093994, 1312.6773853302002, 1248.1251277923584, 1231.8424682617188, 1216.626630783081, 1196.7325172424316, 1132.7993640899658, 1085.4444198608398, 1070.6484031677246, 1048.076005935669, 1026.7085552215576, 1035.3916015625, 1009.8405113220215, 1004.4082088470459, 1031.2165412902832, 978.7646999359131, 918.1333541870117, 922.0090370178223, 948.2602500915527, 915.8232135772705, 889.0281620025635, 893.9188022613525, 856.3256721496582, 855.5097332000732, 851.8275356292725, 863.050145149231, 820.3881130218506, 827.5755319595337, 804.0920095443726, 785.6187515258789, 784.6621618270874, 762.1324834823608, 762.1181087493896, 756.592490196228, 733.3610315322876, 737.6823892593384, 736.2033834457397, 748.8463201522827, 739.1130666732788, 728.5883054733276, 719.2414255142212, 704.6375217437744, 705.5297536849976, 706.7116203308105, 704.6562480926514, 697.0998096466064, 700.4496803283691, 699.0005617141724, 684.7449064254761, 707.8313264846802, 695.5687456130981, 681.2226543426514, 682.3751859664917, 672.9871711730957, 661.0728073120117, 664.4820232391357, 664.2401609420776, 654.8716850280762, 662.3047847747803, 657.3569107055664, 652.6245307922363, 651.6292037963867, 650.1605863571167, 648.1743087768555, 634.5416660308838, 630.414873123169, 631.0431327819824, 625.788688659668, 621.4283227920532, 629.7034769058228, 620.1763954162598, 619.2359762191772, 620.8627510070801, 616.5880165100098, 614.1215171813965, 616.2722606658936, 604.0413885116577, 597.4399309158325, 601.2441740036011, 595.7475309371948, 602.468825340271, 599.6413116455078, 600.7536926269531, 603.5519075393677, 590.4978294372559, 595.9045515060425, 585.7264709472656, 585.5486068725586, 587.7963705062866, 591.5118188858032, 587.1340942382812, 578.8105554580688, 580.2874546051025, 582.6969518661499, 575.4505844116211, 572.1476736068726, 576.1881647109985, 574.4514665603638, 571.0208377838135, 574.8520154953003, 582.5811548233032, 568.720606803894, 560.6892280578613, 561.2641401290894, 562.6148681640625, 565.7577524185181, 565.8452644348145, 563.7986087799072, 558.7273168563843, 567.4673328399658, 560.4637565612793, 564.7814931869507, 563.6651248931885, 560.1137247085571, 556.7260122299194, 561.5038862228394, 559.4932832717896, 561.4488649368286, 565.58899974823, 550.6630754470825, 553.2562885284424, 552.4956979751587, 550.8238544464111, 552.5829086303711, 555.5047426223755, 555.2216815948486, 561.9357223510742, 553.218560218811, 551.7837533950806, 555.1822080612183, 548.3027849197388, 553.430212020874, 551.5245752334595, 549.8936157226562, 550.9418430328369, 551.3300046920776, 543.8067274093628, 544.6434383392334, 546.9850015640259, 547.191897392273, 546.8939952850342, 546.3086957931519, 547.2269697189331, 545.842529296875, 543.7656469345093, 539.8123712539673, 542.8520889282227, 540.1392526626587, 542.1853313446045, 536.7909593582153, 539.3497762680054, 543.3171844482422, 538.3502178192139, 537.7131280899048, 536.0244626998901, 540.8565216064453, 535.4668369293213, 537.082202911377, 540.7897081375122, 536.165641784668, 535.949517250061, 538.8264541625977, 537.3678646087646, 540.5413484573364, 537.1768970489502, 537.9241008758545, 535.5175313949585, 535.0603227615356, 536.030369758606, 534.7364654541016, 533.5303440093994, 538.1338481903076]


    #stl I2cs 98.02/48.43
    #cifar10 I2cs 94.3/70.4
