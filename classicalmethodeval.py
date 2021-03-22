# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:25:54 2020

@author: Administrator
"""
import visdom
from imdataset import ExtendSuperviseddataset
from sklearn import neighbors,cluster,decomposition,svm
import torch as tc
from PIL import Image
import numpy as np
import torch.nn.functional as F
from hwnet import FeatureBackbone,PDhead
from sklearn.metrics import confusion_matrix
import time
from sklearn import metrics
traindataset=ExtendSuperviseddataset(sampletype=0)

testdataset=ExtendSuperviseddataset(sampletype=2)

def transdata(ds):
    xx=tc.tensor([])
    yy=tc.tensor([])
    for x,y in ds:
        xx=tc.cat([xx,x.view(1,-1)])
        yy=tc.cat([yy,y.flatten()])
    return xx, tc.where(yy>0.5,tc.tensor(1.0),tc.tensor(0.0))


def nncls(x,y,tx,ty=None):
    #x,y are train pairs 
    #tx,ty are test pairs
    cls=neighbors.KNeighborsClassifier(n_neighbors=11)
    cls.fit(x,y)  #0.959
    if ty is not None:
        print('the precision of nn clsifier is',cls.score(tx,ty))
    return cls.predict(tx)  
def nncls1(x,y,tx,ty=None):
    #x,y are train pairs 
    #tx,ty are test pairs
    cls=neighbors.KNeighborsClassifier(n_neighbors=11)
    cls.fit(x,y)  #0.959
    if ty is not None:
        print('the precision of nn clsifier is',cls.score(tx,ty))
    
    st=time.time()
    r =cls.predict(tx)
    ctime=time.time()-st
    label=ty
    fpr, tpr, thresholds = metrics.roc_curve(label, r)
    auc=metrics.auc(fpr, tpr)
       
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
    return r 

def pcanncls(x,y,tx,ty=None):
    pca=decomposition.PCA(n_components=30)
    x_=pca.fit_transform(x)
    tx_=pca.transform(tx)    #0.967
    return nncls(x_,y,tx_,ty)
def pcanncls1(x,y,tx,ty=None):
    pca=decomposition.PCA(n_components=30)
    x_=pca.fit_transform(x)
    tx_=pca.transform(tx)    #0.967
    return nncls1(x_,y,tx_,ty)


def otsu(x,y,tx,ty=None):
    cls=cluster.KMeans(n_clusters=2)
    cls.fit(x[:,544:545])
    threshold=cls.cluster_centers_.mean()
    px=tx[:,544:545]
    pt=tc.where(px>threshold,tc.tensor(1.0),tc.tensor(0.0) ).flatten()
    if ty is not None:
        r=(pt-ty).abs().sum().item()   #0.899
        print('the precision of otsu is', 1-r/tx.shape[0])
    return pt

def otsu1(x,y,tx,ty=None):
    cls=cluster.KMeans(n_clusters=2)
    cls.fit(x[:,544:545])
    threshold=cls.cluster_centers_.mean()
    px=tx[:,544:545]
    st=time.time()
    r=tc.where(px>threshold,tc.tensor(1.0),tc.tensor(0.0) ).flatten()
    ctime=time.time()-st
    label=ty
    fpr, tpr, thresholds = metrics.roc_curve(label, r)
    auc=metrics.auc(fpr, tpr)
       
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
    
    if ty is not None:
        r=(r-ty).abs().sum().item()   #0.899
        print('the precision of otsu is', 1-r/tx.shape[0])
    return r

def svmcls(x,y,tx,ty=None):
    cls=svm.SVC(kernel='rbf',gamma='auto',C=2)#kernel='linear')
    cls.fit(x,y)      #0.963
    if ty is not None:
        print('the precision of svm clsifier is',cls.score(tx,ty))
    return cls.predict(tx)
def svmcls1(x,y,tx,ty=None):
    cls=svm.SVC(kernel='rbf',gamma='auto',C=2)#kernel='linear')
    cls.fit(x,y)      #0.963
    if ty is not None:
        print('the precision of svm clsifier is',cls.score(tx,ty))
    st=time.time()
    r =cls.predict(tx)
    ctime=time.time()-st
    label=ty
    fpr, tpr, thresholds = metrics.roc_curve(label, r)
    auc=metrics.auc(fpr, tpr)
       
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
    return r


def dingmethod2(tx,ty,testx,testy,threshold=-0.05):
    testx=testx.view(-1,1,33,33)
    xw=tc.zeros((3,3))
    xw[1,0]=1.0
    xw[1,-1]=-1.0
    yw=xw.T.clone()
    xw=xw.view((1,1,3,3))
    yw=yw.view((1,1,3,3))
    dx=F.conv2d(testx,xw,padding=1).abs_()
    dy=F.conv2d(testx,yw,padding=1).abs_()
    w=tc.where(dx>dy,dx,dy)
    meankernel=tc.ones((1,1,3,3))
    T=F.conv2d(w*testx,meankernel,padding=1)  
    r=testx-T
    r1=tc.where(r<tc.tensor(threshold),tc.tensor(1.0),tc.tensor(0.0))
    r2=F.pad(r1,(-16,-16,-16,-16))
    pt=r2.view(-1)
    r=(pt-testy).abs().sum().item()   #0.951
    print('the precision of otsu is', 1-r/pt.shape[0])
    return pt

def dingmethod21(tx,ty,testx,testy,threshold=-0.05):
    testx=testx.view(-1,1,33,33)
    xw=tc.zeros((3,3))
    xw[1,0]=1.0
    xw[1,-1]=-1.0
    yw=xw.T.clone()
    xw=xw.view((1,1,3,3))
    yw=yw.view((1,1,3,3))
    st=time.time()
    dx=F.conv2d(testx,xw,padding=1).abs_()
    dy=F.conv2d(testx,yw,padding=1).abs_()
    w=tc.where(dx>dy,dx,dy)
    meankernel=tc.ones((1,1,3,3))
    T=F.conv2d(w*testx,meankernel,padding=1)  
    r=testx-T
    r1=tc.where(r<tc.tensor(threshold),tc.tensor(1.0),tc.tensor(0.0))
    r2=F.pad(r1,(-16,-16,-16,-16))
    r=r2.view(-1) 
    ctime=time.time()-st
    label=testy
    fpr, tpr, thresholds = metrics.roc_curve(label, r)
    auc=metrics.auc(fpr, tpr)
       
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
    
    if ty is not None:
        r=(r-testy).abs().sum().item()   #0.899
        print('the precision of otsu is', 1-r/tx.shape[0])
    return r


def dingmethod(imgpath='./segtest/1591536211.bmp',threshold=-0.05):
    t=tc.from_numpy(np.array(Image.open(imgpath).convert('L')))
    t=t.unsqueeze(0).unsqueeze(0)/255.0
    insize=t.shape[-1]
    print('input image shape',insize)
    xw=tc.zeros((3,3))
    xw[1,0]=1.0
    xw[1,-1]=-1.0
    yw=xw.T.clone()
    xw=xw.view((1,1,3,3))
    yw=yw.view((1,1,3,3))
    print(t.shape,xw.shape)
    dx=F.conv2d(t,xw,padding=1).abs_()
    dy=F.conv2d(t,yw,padding=1).abs_()
    w=tc.where(dx>dy,dx,dy)
    meankernel=tc.ones((1,1,3,3))
    T=F.conv2d(w*t,meankernel,padding=1)  
    r=t-T
    r1=tc.where(r<tc.tensor(threshold),tc.tensor(255.0),tc.tensor(0.0))
    r2=F.pad(r1,(-16,-16,-16,-16))
    r3=F.pad(r2,(16,16,16,16),value=128)
    return r3
def runding():
    import glob
    import os
    imgs=glob.glob('./segtest/*.bmp')
    for im in imgs:
        r=dingmethod(im,-0.05)
        img=Image.fromarray(r.byte().squeeze(0).squeeze(0).numpy(),'L')
        img.save('./tmp/ding/'+os.path.basename(im))
        

def unfoldimage(imgpath='./segtest/1591536211.bmp'):
    t=tc.from_numpy(np.array(Image.open(imgpath).convert('L')))
    t=t.unsqueeze(0).unsqueeze(0)/255.0
    insize=t.shape[-1]
    print('input image shape',insize)
    uf=tc.nn.Unfold((33,33))
    ot=uf(t)
    ot.squeeze_(0).t_()
    return ot

def foldimage(res,outpath='./tmp/1.png'):
    if tc.is_tensor(res):
        pass
    else:
        res=tc.from_numpy(res)
    imgsize=int((res.shape[0])**0.5)
    res=res.reshape((imgsize,imgsize))
    res=res*255.0
    r=F.pad(res,(16,16,16,16),value=128)
    img=Image.fromarray(r.byte().numpy(),'L')
    img.save(outpath)
   
########################
def hwnet(c=979):
    device='cuda'
    fb =tc.load('./tmp/ssdwd0.001 wd0.005 ssdn30/fbp{}_1'.format(c)).to(device)  
    ssh =tc.load('./tmp/ssdwd0.001 wd0.005 ssdn30/pdh{}_1'.format(c)).to(device)  
    valid_loader=tc.utils.data.DataLoader( 
        ExtendSuperviseddataset(sampletype=2),
        batch_size=50, shuffle=False,drop_last=False)
    fb.eval()
    ssh.eval()
    totalnum=0
    correctnum=0
    lblist=[]
    rlist=[]
    crlist=[]
    for batch_idx, (datax, datay) in enumerate(valid_loader):
        data, label = datax.to(device), datay.to(device)
        f = fb(data)
        r=ssh(f).cpu()
        label=tc.where(label.cpu()>tc.tensor(0.5),tc.tensor(1.0),tc.tensor(0.0))
        cr=tc.where(r>tc.tensor(0.5),tc.tensor(1.0),tc.tensor(0.0))
        lblist+=label.view(-1).tolist()
        rlist+=r.view(-1).tolist()
        crlist+=cr.view(-1).tolist()
        #for i in range(50):
        #    if abs(lblist[-50+i]-crlist[-50+i])>0.5:
         #       vis.surf(data[i],opts={'xmax':1,'title':'t{}p{}pb{}'.format(int(lblist[-50+i]),int(crlist[-50+i]),rlist[-50+i])})
    return lblist,crlist,rlist #ground true, predict class, predict probability
  
def hwnet1(c=973):
    device='cuda'
    fb =tc.load('./tmp/fromscrach wd0.005/fbp{}'.format(c)).to(device)  
    ssh =tc.load('./tmp/fromscrach wd0.005/pdh{}'.format(c)).to(device)  
    valid_loader=tc.utils.data.DataLoader( 
        ExtendSuperviseddataset(sampletype=2),
        batch_size=1000, shuffle=False,drop_last=False)
    fb.eval()
    ssh.eval()
    for batch_idx, (datax, datay) in enumerate(valid_loader):
        data, label = datax.to(device), datay.to(device)
        tc.cuda.synchronize()
        start=time.time()
        f = fb(data)
        r=ssh(f)
        tc.cuda.synchronize()
        ctime = time.time()-start
        r=r.cpu().view(-1)
        
        label=(label.view(-1).cpu()>0.5)*1.0
        
        fpr, tpr, thresholds = metrics.roc_curve(label.detach().numpy(), r.detach().numpy())
        auc=metrics.auc(fpr, tpr)
        #ac=metrics.accuracy_score(label.detach().numpy(), r.detach().numpy()>0.5)
        tn, fp, fn, tp = metrics.confusion_matrix(label.detach().numpy() ,
                                                  r.detach().numpy()>0.5).ravel()
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
            
    
    return None #ground true, predict class, predict probability
  



if __name__=='__main__':
    import glob
    import os
    #vis=visdom.Visdom()
    try:
        tx,ty,testx,testy=tc.load('./trainxy_testxy.pt') 
        print('load from datafile')
    except Exception:
        print('load data from raw')
        tx,ty=transdata(traindataset)
        testx,testy=transdata(testdataset)
    imgs=glob.glob('./segtest/*.bmp')
    
    for im in imgs:
        #testx=unfoldimage(im)
        #py=nncls(tx,ty,testx)
        py=pcanncls(tx,ty,testx)
        #py=otsu(tx,ty,testx)
        #py=svmcls(tx,ty,testx)
        foldimage(py,'./tmp/predict/'+os.path.basename(im))