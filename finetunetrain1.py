
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imdataset1 import ExtendSuperviseddataset
from hwnet import FeatureBackbone,PDhead
import time
from torchvision import transforms
from PIL import Image
import torch as tc
from sklearn import metrics

def train(args, fb,ssh, device, train_loader, fboptimizer,sshoptimizer, epoch):
    fb.train()
    ssh.train()
    for batch_idx, (datax, datay) in enumerate(train_loader):
        data, label = datax.to(device), datay.to(device)
        
        fboptimizer.zero_grad()
        sshoptimizer.zero_grad()
        
        f = fb(data)
        r=ssh(f)
        
        loss = nn.BCELoss( )(r,label)
        loss.backward()

        fboptimizer.step()
        sshoptimizer.step()
      
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
def test(args, fb,ssh, device, test_loader):
    fb.eval()
    ssh.eval()    
    
    for batch_idx, (datax, datay) in enumerate(test_loader):
        data, label = datax.to(device), datay.to(device)
        tc.cuda.synchronize()
        start = time.time()
        f = fb(data)
        r=ssh(f)
        tc.cuda.synchronize()
        ctime = time.time()-start
        r=r.cpu().view(-1)
        
        label=(label.view(-1).cpu()>0.5)*1.0
        
        fpr, tpr, thresholds = metrics.roc_curve(label.detach().numpy(), r.detach().numpy())
        auc=metrics.auc(fpr, tpr)
        ac=metrics.accuracy_score(label.detach().numpy(), r.detach().numpy()>0.5)
    if (ac>0.5):
        print('accuracy: {}'.format(ac))
        print('auc:{}'.format(auc))
        print('time for reference:{}'.format(ctime))
        
    print("there are total {} test samples, correct classfied {} samples".format(1000,1000*ac))
    if (ac>1):
        correctnum=int(ac*1000)
        torch.save(fb,'./tmp/fbp{}'.format(correctnum ))
        torch.save(ssh,'./tmp/pdh{}'.format(correctnum))   
        

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Scratch segmentation Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
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
    fbst=100
    fb =torch.load('./tmp/fb{}'.format(fbst)).to(device)  if fbst else FeatureBackbone(1,96).to(device)  
    ssh =torch.load('./tmp/pdh{}'.format(startepoch)).to(device)  if startepoch else PDhead().to(device)
    
    args.epochs=100
    #disoptimizer = optim.Adam(dismodel.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 
    fboptimizer=optim.SGD(fb.parameters(), lr=args.lr*0.01, momentum=args.momentum)
    sshoptimizer=optim.SGD(ssh.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=0.005)
    
        
    for epoch in range(startepoch+1, startepoch+args.epochs + 1):
        train(args, fb,ssh, device, train_loader, fboptimizer, sshoptimizer,epoch)
        test(args, fb,ssh, device, valid_loader)
        time.sleep(10)
        if epoch%10==0:
            print(epoch)
            #torch.save(fb,'./tmp/fbp{}'.format(epoch))
            #torch.save(ssh,'./tmp/pdh{}'.format(epoch))
    
if __name__ == '__main__':
    main()
