from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageEnhance
from torchvision import transforms
import glob
import torch as tc 
import numpy as np
import random

class SSDataset(Dataset):
    '''a rotation flip transformation dataset for self-supervised learning.  '''
    def __init__(self,imgdir='./Camera/*.bmp',numberofsample=50,patchsize=(33,33),imgmode='L'):
        ''' imgdir is the direction stores the large images;
            numberofsample is the sample patch is cliped from each image;
            patchsize is the clipped size a tuple indicate the (height,width)  currently only support square shape which means width==height 
            imgmode is the output tensor type, 'L' means 1 channel grey image, 'RGB' means 3 channel rgb image, only these to choise are available currently.

            how this is function: when getitem ,will clip a random region from image in sequence
        '''
        self.imgdir=imgdir
        self.eachnumberaimage=numberofsample
        self.patchsize=patchsize 
        self.imgmode=imgmode
        
        self.fns=glob.glob(self.imgdir)
        self.imgfilelength=len(self.fns)
    def __len__(self):
        return self.imgfilelength*self.eachnumberaimage
    def __getitem__(self,idx):
        if idx>=self.__len__():
            raise IndexError
        imgid=idx%self.imgfilelength
        imgxname=self.fns[imgid]
        image = Image.open(imgxname).convert(self.imgmode)

        #toto randomsample is not a clever it should guided the postion to crop is more efficient I think
        maxstd=0
        tmpsample=None
        for i in range(random.randint(1,30)):
            sampleimg=transforms.RandomCrop(self.patchsize)(image)
            std,mean=tc.std_mean(transforms.ToTensor()(sampleimg))
            if std>maxstd:
                maxstd=std
                tmpsample=sampleimg
        sampleimg=tmpsample if tmpsample else sampleimg
        #do random rotation
        rotationtypes=['0',Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270]
        transtype=random.choice(rotationtypes)
        
        if transtype !='0':
            patchx = sampleimg.transpose(transtype)
        else:
            patchx=sampleimg
        
        patchxtensor=transforms.ToTensor()(patchx)    #to tensor  c x h x w   -0.5 --- 0.5
        
        patchy = transforms.RandomVerticalFlip(p=0.5)(sampleimg)
        patchy = transforms.RandomHorizontalFlip(p=0.5)(patchy)
        patchytensor=transforms.ToTensor()(patchy)

        return patchxtensor,patchytensor    #returns a random patch

class ExtendSuperviseddataset(Dataset):
    '''supervised dataset'''
    def __init__(self,sampletype=0,balanced=True,number=[3000,400,1000]):
        self.number=number
        self.length=number[sampletype]
        import os
        
        self.labelfile='./extendlabeled/labels.txt'
        self.imagefiles=open(self.labelfile).readlines()
        
        self.bias=0
        self.changeset(sampletype)
    def changeset(self,sampletype=0):
        '''sampletype is 0,1,2, 0 is training set, 1 is validation set, 2 is test set
        '''
        self.length=self.number[sampletype]
        if sampletype==0:
            self.bias=0
        elif sampletype==1:
            self.bias=self.number[0]
        else:
            self.bias=self.number[0]+self.number[1]
        
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        if idx>=self.length:
            raise IndexError
        idx+=self.bias
        
        transtypes=[Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM,
                Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270]
        #transtype=transtypes[np.random.randint(len(transtypes))]
        transtype=random.choice(transtypes)
        
        imgname,label=self.imagefiles[idx].strip().split(',')
        
        samples = Image.open(imgname).convert('L')
        if self.length>1000:
            samples = samples.transpose(transtype)
        sampletensor=transforms.ToTensor()(samples)
        return sampletensor,tc.Tensor( [[[0.9]]]) if float(label)>0.5 else tc.Tensor( [[[0.1]]])  #label smoothing
        
        '''if idx%2==1:#particle
            particleimg = Image.open(self.particlefiles[self.xidx[idx//2]]).convert('L')
            particleimg = particleimg.transpose(transtype)
            particletensor=transforms.ToTensor()(particleimg)
            return particletensor,tc.Tensor( [[[0.9]]])   #label smoothing
        else : #nonparticle
            nonparticleimg = Image.open(self.nonparticlefiles[self.yidx[idx//2]]).convert('L')
            nonparticleimg = nonparticleimg.transpose(transtype)
            nonparticleimg=transforms.ToTensor()(nonparticleimg)
            return nonparticleimg,tc.Tensor( [[[0.1]]])
        
        '''

if __name__=='__main__':
    import visdom 
    vis=visdom.Visdom()
    #sd=SSDataset()
    sd=ExtendSuperviseddataset()
    s1,s2=sd[0]
    vis.surf(s1)
    #vis.surf(s2)
    '''d=ImDataset(istrain=True)
    idx=random.randint(0,20)
    x,y=d[idx]
    print(x.shape)
    print(y.shape)
    transforms.ToPILImage()(x).save('./tmp/x.png')
    transforms.ToPILImage()(y).save('./tmp/y.png')'''
  
