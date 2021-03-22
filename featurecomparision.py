# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:35:42 2021

@author: Administrator
"""

import torch as tc 
from hwnet import FeatureBackbone
from sklearn import manifold
from matplotlib import pyplot as plt








def tsne(fts,lb):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(fts.view(-1,96).detach().numpy())
    c1=X_tsne[lb>=0.5]
    c2=X_tsne[lb<0.5]
    
    fig, ax = plt.subplots()
    
    x=c1[:,0]
    y=c1[:,1]
    ax.scatter(x, y, c='tab:blue', s=15, label='Particle',
                       alpha=1, edgecolors='none')
    x=c2[:,0]
    y=c2[:,1]
    ax.scatter(x, y, c='tab:red', s=15, label='None-particle',
                       alpha=1, edgecolors='none')
    
    ax.legend()
    ax.grid(True)

    plt.show()

def drawselftrainedfeature1():
    fbtrained=tc.load('./tmp/ssdwd0.001 wd0.005 ssdn30/fbp{}_1'.format(979)).to('cpu') 
    tsne(fbtrained(testxx),testy)
def drawselftrainedfeature():
    fbtrained=tc.load('./tmp/ssdwd0.001 wd0.005 ssdn30/fbp{}'.format(979)).to('cpu') 
    tsne(fbtrained(testxx),testy)

def drawraw():
    fbraw= FeatureBackbone(1,96)
    tsne(fbraw(testxx),testy)
    
def drawfeaturefromscratch():
    fbtrainedraw =tc.load('./tmp/fromscrach wd0.005/fbp{}'.format(973)).to('cpu') 
    tsne(fbtrainedraw(testxx),testy)

def drawtest(n):
    fbtrainedraw =tc.load('./tmp/fb{}'.format(n)).to('cpu') 
    tsne(fbtrainedraw(testxx),testy)
if __name__=='__main__':
    tx,ty,testx,testy=tc.load('./trainxy_testxy.pt') 
    testxx=testx.view(1000,1,33,33)
    #drawfeaturefromscratch()
    #drawraw()
    #drawselftrainedfeature()
    drawtest(0)
    
