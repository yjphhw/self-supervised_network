import torch as tc
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0,bias=False),
            nn.InstanceNorm2d(out_ch),  
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=0,bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class FeatureBackbone(nn.Module):
    '''This is F in paper'''
    def __init__(self,in_ch,out_ch,size=33):
        super(FeatureBackbone, self).__init__()
        self.conv1=DoubleConv(1,16)
        self.conv2=DoubleConv(16,16)
        self.conv3=DoubleConv(16,16)
        self.conv4=DoubleConv(16,16)
        self.conv5=DoubleConv(16,16)
        self.conv6=DoubleConv(16,16)
        self.conv7=DoubleConv(16,16)
        self.conv8=nn.Conv2d(16, out_ch, 3, padding=0)
        self.bn8=nn.InstanceNorm2d(out_ch)
        self.r8=nn.ReLU(inplace=True)
        self.conv9=nn.Conv2d(out_ch, out_ch, 3, padding=0)
        self.size=size
        self.sizes=[self.size-6-4*i for i in range(6) ]
        
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)+x[:,:,2:self.sizes[0],2:self.sizes[0]]
        x=self.conv3(x)+x[:,:,2:self.sizes[1],2:self.sizes[1]]
        x=self.conv4(x)+x[:,:,2:self.sizes[2],2:self.sizes[2]]
        x=self.conv5(x)+x[:,:,2:self.sizes[3],2:self.sizes[3]]
        x=self.conv6(x)+x[:,:,2:self.sizes[4],2:self.sizes[4]]
        x=self.conv7(x)+x[:,:,2:self.sizes[5],2:self.sizes[5]]
        x=self.conv8(x)
        x=self.bn8(x)
        x=self.r8(x)
        x=self.conv9(x)
        return x
    

class SShead(nn.Module):
    '''This is G in paper '''
    def __init__(self,indimension=96):
        super(SShead,self).__init__()
        self.l1=nn.Linear(indimension,96)
        self.d1=nn.Dropout(0.2)
        self.l2=nn.Linear(96,1)
    def forward(self,x):
        x=x.flatten(1)
        x=tc.sigmoid(self.l1(x))
        #x=self.d1(x)
        x=self.l2(x)
        return x

class PDhead(nn.Module):
    '''This is C in paper '''
    def __init__(self):
        super(PDhead,self).__init__()
        self.l1=nn.Conv2d(96, 48, 1)  #nn.Linear(96,48)
        self.d1=nn.Dropout(0.2)
        self.l2=nn.Conv2d(48,1,1)
    def forward(self,x):
        x=tc.sigmoid(self.l1(x))
        x=self.d1(x)
        x=tc.sigmoid(self.l2(x))
        return x

if __name__=='__main__':
    import imdataset
    d=imdataset.ImDataset(istrain=True)
    fb=FeatureBackbone(1,96) #
    ssh=SShead() #
    ssh.eval()
    import torch as tc
    x,y=d[12]
    i=tc.cat((x,y),0).unsqueeze(1)
    
    r=ssh(fb(i))
    print(r)
    loss=tc.abs(r[0,0]-r[1,0])
    print(loss)
    #fb=FeatureBackbone(1,96,512+32)
    fb.sizes=[56+32-6-4*i for i in range(6) ]
    d=tc.rand(1,1,56+32,56+32)
    pd=PDhead()
    r=pd(fb(d))
    print(r.shape)
