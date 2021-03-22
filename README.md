# A Self-supervised CNN for Particle Inspection on Optical Element

Networks are implemented in hwnet.py. The structure of F, G, and C in this file.

trainselfsupservisednetwork.py is used to train Self-supervised network. 

finetunetrain1.py is used to finetune the network with F transferred from self-supervised network

classicalmethodeval.py  implements classical methods for comparison and do evaluation of the proposed model.

deeplearningbasedmethod.py implements deep learning methods for comparison.

featurecomparison.py is used for feature visualization by t-SNE method.

threeplot is used for creating 3D profile.

imdataset1 defines the self-supervised dataset and labeled dataset.

2019_02_28_16_38_27.zip is one whole image taken by instrument, and is used for generate samples for training self-supervised network.

extendlabeled.zip contains labeled 4,400 samples. generateidx.py should be execute before loading samples by imdataset1.

trainxy_testxy.pt is samples in pytorch tensorfile for easy use and is same to samples in extendlabeled.zip
