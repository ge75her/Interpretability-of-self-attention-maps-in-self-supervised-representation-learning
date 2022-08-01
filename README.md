# Interpretable-representation-learning-with-self-supervised-learning
This repository is utilized to identify the factors that influent the attention map with self-supervised learning.
## Create Dataset
- download `./data/mnist.pkl.gz` dataset. This is the original MNIST dataset. It contains 50,000 train images, 10,000 test images and 10,000 validation images <br>
- run `dataset.ipynb` file to get the cluttered_mnist dataset. To make the result more intuitive, convert .npz dataset to .JPG format <br>
## Pretrained models on PyTorch Hub
Pretrained model is used to speed up the training process. The evaluation results of different models can be found from [DINO](https://github.com/facebookresearch/dino) <br>
```
import torch <br>
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16') <br>
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8') <br>
vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16') <br>
vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8') <br>
xcit_small_12_p16 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16') <br>
xcit_small_12_p8 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p8') <br>
xcit_medium_24_p16 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16') <br>
xcit_medium_24_p8 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8') <br>
resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
```
## Training
most code are copy from [DINO](https://github.com/facebookresearch/dino) <br>
- run `train.py` or `whole_pipline.ipynb` to training the dataset, save the output results to `./logs` folder <br>
- run `visual_augmentation.ipynb` to visualize the data augmentation results <br>
- after training process, run `visual_attention.ipynb` to visualize the final attention map <br>
## Result
TODO
- Background
- resolution
- simCLR loss VS DINO loss
- multicrop
- size of attention map
