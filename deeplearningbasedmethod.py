
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imdataset3band import Superviseddataset,ExtendSuperviseddataset
from hwnet import FeatureBackbone,PDhead
import time
from torchvision import transforms
from PIL import Image
import torch as tc
from sklearn import metrics
import torchvision.models as models
import time
def getRes18net():
    resnet18 = models.resnet18()
    resnet18.fc=nn.Linear(512,1,True)
    resnet18.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return  resnet18.to('cuda')

def mobilenetv2():
    mobilenetv2 = models.mobilenet_v2()
    mobilenetv2.features[0][0]=nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    mobilenetv2.classifier[1]=nn.Linear(1280,1,True)
    return mobilenetv2.to('cuda')

def vgg11():
    vgg11=models.vgg11()
    vgg11.features[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    vgg11.classifier[6]=nn.Linear(4096,1,True)
    return vgg11.to('cuda')

def shufflenet():
    sf1=models.shufflenet_v2_x0_5()
    sf1.conv1[0]=nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    sf1.fc=nn.Linear(1024,1,True)
    return sf1.to('cuda')
def train(args, model, device, train_loader,optimizer, epoch):
    model.train()
   
    for batch_idx, (datax, datay) in enumerate(train_loader):
        data, label = datax.to(device), datay.to(device)
        
        optimizer.zero_grad()
        
        r = model(data)
        r=tc.sigmoid(r).view(-1)
        #print(r.shape, label.shape)
        loss = nn.BCELoss( )(r,label.view(-1))
        loss.backward()

        optimizer.step()
        
        if (batch_idx+1) % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            #print(r.flatten(),label.flatten())
            time.sleep(7)
    '''if epoch%2==0:
        imgx=transforms.ToPILImage()(output.detach()[0,:,:,:].cpu())
        imgxx=transforms.ToPILImage()(data.detach()[0,:,:,:].cpu()) 
        imgx.save("./tmp/{}predict.bmp".format(epoch))
        imgxx.save('./tmp/{}input.bmp'.format(epoch))
        imgimc=transforms.ToPILImage()(cim[0]) 
        imgimc.save('./tmp/{}real.bmp'.format(epoch))
        '''
def test(args, model, device, test_loader):
    model.eval()
    totalnum=0
    correctnum=0
    for batch_idx, (datax, datay) in enumerate(test_loader):
        start = time.time()
        data, label = datax.to(device), datay.to(device)
        tc.cuda.synchronize()
        
        r = model(data)
        tc.cuda.synchronize()
        ctime = time.time()-start
        r=r.cpu().view(-1)
        
        label=(label.view(-1).cpu()>0.5)*1.0
        label=label.detach().numpy()
        r=r.detach().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(label, r)
        auc=metrics.auc(fpr, tpr)
        #ac=metrics.accuracy_score(label,r>0.5)
        tn, fp, fn, tp = metrics.confusion_matrix(label ,
                                                  r>0.5).ravel()
        ac=(tp+tn)/(tn+fp+fn+tp)
        tpr=tp/(tp+fn)
        fpr=fp/(fp+tn)
        if (ac>0.5):
            print('accuracy: {}'.format(ac))
            print('tpr:{}'.format(tpr))
            print('fpr:{}'.format(fpr))
            print('auc:{}'.format(auc))
            print('time for reference:{}'.format(ctime))
 
    print("there are total {} test samples, correct classfied {} samples".format(1000,1000*ac))
        
        

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Scratch segmentation Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                        help='SGD momentum (default: 0.95)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :',device)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader( 
        ExtendSuperviseddataset(sampletype=0),
        batch_size=args.batch_size, shuffle=True,drop_last=False, **kwargs)
    valid_loader=torch.utils.data.DataLoader( 
        ExtendSuperviseddataset(sampletype=2),
        batch_size=1000, shuffle=False,drop_last=False, **kwargs)

    startepoch=0
   
    #model=shufflenet() 
    #model=vgg11() 
    #model=mobilenetv2() #getRes18net()
    model=getRes18net()
    args.epochs=7
    optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=0.005)
    
        
    for epoch in range(startepoch+1, startepoch+args.epochs + 1):
        train(args, model, device, train_loader, optimizer,epoch)
        test(args, model, device, valid_loader)
        #time.sleep(10)
        if epoch%10==0:
            print(epoch)
            
    
if __name__ == '__main__':
    main()
