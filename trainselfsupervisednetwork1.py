import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imdataset1 import SSDataset
from hwnet import FeatureBackbone,SShead
import time
from torchvision import transforms
from PIL import Image
import torch as tc

def train(args, fb,ssh, device, train_loader, fboptimizer,sshoptimizer, epoch):
    fb.train()
    ssh.train()
    for batch_idx, (datax, datay) in enumerate(train_loader):
        datax, datay = datax.to(device), datay.to(device)
        data=tc.cat((datax,datay),0)

        fboptimizer.zero_grad()
        sshoptimizer.zero_grad()
        
        f = fb(data)
        r=ssh(f)
        loss=tc.abs(r[0,0]-r[1,0])
        if (loss.item()>0.005):
            loss.backward()
            fboptimizer.step()
            sshoptimizer.step()
        else:
            continue
        if batch_idx % args.log_interval == 0:
            print((f[0,0]-f[0,1]).abs().sum())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tOutput: {},{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),r[0,0],r[1,0]) )
   
def test(args, model, device, test_loader,epoch):
    model.eval()
    count=0
    sum_psnr=0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #output=torch.clamp(output,-0.5,0.5)+0.5  #restrick to 0-1
            sum_psnr+=10*torch.log10(1/nn.MSELoss()(target.to(device),output)).item()
            count+=1
    imgx=transforms.ToPILImage()(output.detach()[0,:,:,:].cpu())
    imgxx=transforms.ToPILImage()(data.detach()[0,:,:,:].cpu()) 
    imgx.save("./tmp/{}predict.bmp".format(epoch))
    imgxx.save('./tmp/{}input.bmp'.format(epoch))
    imgimc=transforms.ToPILImage()(target[0].cpu()) 
    imgimc.save('./tmp/{}real.bmp'.format(epoch))
    print('\nTest set: Average psnr: {:.4f}, count is {}\n'.format(sum_psnr/count,count))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='self-supervised Rotation invarient feature learning model')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :',device)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader( 
        SSDataset(),
        batch_size=args.batch_size, shuffle=True,drop_last=False, **kwargs)

    

    startepoch=0
    fb =torch.load('./tmp/fb{}'.format(startepoch))  if startepoch else FeatureBackbone(1,96).to(device)  
    ssh =torch.load('./tmp/ssh{}'.format(startepoch))  if startepoch else SShead(indimension=96).to(device)
    
    
    args.epochs=100
    #disoptimizer = optim.Adam(dismodel.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 
    fboptimizer=optim.SGD(fb.parameters(), lr=args.lr, momentum=args.momentum)
    sshoptimizer=optim.SGD(ssh.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=0.001)
    
        
    for epoch in range(startepoch, args.epochs + 1):
        if epoch%5==0:
            print("save parameters to file in {}.".format(epoch))
            torch.save(fb,'./tmp/fb{}'.format(epoch))
            torch.save(ssh,'./tmp/ssh{}'.format(epoch))
        train(args, fb,ssh, device, train_loader, fboptimizer, sshoptimizer,epoch)
        
    
if __name__ == '__main__':
    main()
