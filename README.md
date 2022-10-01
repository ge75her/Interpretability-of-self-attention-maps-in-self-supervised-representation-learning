# Interpretability of self-attention maps in self-supervised representation learning
Using self-supervised learning methods on binary iamges, this repository is used to determine the variables that influence the self-attention map and to try to determine what are the important components to obtaining a high resolution and high quality self-attention maps. To learn more, please read the essay.
## Create Dataset
We will training on cluttered MNIST dataset. You can setup the dataset by following steps:
- download `./data/mnist.pkl.gz` dataset. This is the original MNIST dataset. It contains 50,000 train images, 10,000 test images and 10,000 validation images <br>
- run `dataset.ipynb` file to get the cluttered_mnist dataset. To make the result more intuitive, convert .npz dataset to .JPG format <br>
## Pretrained models on PyTorch Hub
Pretrained model is used to speed up the training process. The evaluation results of different models can be found from [DINO](https://github.com/facebookresearch/dino) <br>
```
import torch 
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16') 
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8') 
vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16') 
vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8') 
xcit_small_12_p16 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16') 
xcit_small_12_p8 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p8') 
xcit_medium_24_p16 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16') 
xcit_medium_24_p8 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8') 
resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
```
## Training
most code are copy from [DINO](https://github.com/facebookresearch/dino) <br>
- run `train.py` or `whole_pipline.ipynb` to training the dataset, save the output results to `./logs` folder <br>
- run `visual_augmentation.ipynb` to visualize the data augmentation results <br>
- after training process, run `visual_attention.ipynb` to visualize the final attention map <br>
## Result
### Multi-crops traning
multi-crops training is an important component for self-supervised ViT pretraining. Comparing with 4 crops per branch, 2 crops per branch and one single crop per branch, the resulting findings are not significantly different. There may not be much information in the binary images, which is one potential reason.
### Loss Function
NCE(Noise Contractive Estimator) loss function and Cross Entropy loss function are two commonly loss function in contrastive learning. As a result, training with both
of them can predict a quite well classification as shown in self-attention map. The loss function is not an important component that influent self-attention map.
### Data Augmentation
we fed different views to two branches. The results show that for binary images, the size of the input images and the different views we fed into the model are two important components to obtain a high resolution and high quality self-attention maps. However, the performance on binary images is not as good as on color images. 
## Future work
- training on a bigger batch size
- investigate the meaning of different heads on ViT, try to figure out how many head do we really need
- consider other state-of the art contrastive learning methods
