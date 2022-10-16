# Train K-net on COCO2017

## Introduction 

Using semantic segmentation, we can identify each part accurately using the Unet model with MobilenetV2 as the backbone. This method is much more efficient in identifying car parts through images than object detection methods. The efficiency stems from the fact that each pixel belongs to only one. In object detection, on the other hand, because of overlapping boxes, each pixel can belong to multiple part classes. Another benefit is that the detection box can contain the background part which is almost removed in the segmentation model.


## About K-net

Although Unet does give reasonable predictions, there are cases where the model mis-performs in test cases of car images. It may provide an incorrect mask for highly reflective, glossy cars, the boundary may not be smooth, minor parts may go undetected, or it may deliver a patchy mask or broken mask. So, there is a need to devise a model largely free from these defects. One such model we can use is Knet.

Knet is a unified framework for semantic Segmentation, instance segmentation, and panoptic Segmentation. The K in Knet stands for kernels, which segments semantic and instance categories through a group of kernels. Each kernel is responsible for generating a mask, either for instance or stuff class. In the previous architecture, we had static kernels operate on image and feature maps, but this time we have dynamic kernels. The randomly initialized kernels train upon the targets they seek to predict. Semantic kernels predict semantic categories, while instance kernels predict the instance objects from the image. This simple combination of semantic and instance kernels allows us to make panoptic segmentation which combines both semantic and instance segmentation to provide richer segmentation results with a better understanding of context.


![pasted image 0](https://user-images.githubusercontent.com/78655282/196053976-72369d49-6177-41db-b9ac-8c9c1ff6b177.png)


As the figure above shows, the semantic kernels produce the semantic masks. The semantic masks are based on the class of object pixels they belong to. Multiple objects belonging to the same class are allocated a single mask in case of semantic segmentation.


![pasted image 0 (2)](https://user-images.githubusercontent.com/78655282/196053988-1bacae5c-e210-4435-99b9-1f1607ea5ce4.png)


The picture above shows the instance kernels operating on feature maps to produce instance masks. Instance masks make one mask for a single object. Multiple masks are created if there are multiple objects of the same class, which is the essence of instance segmentation.


![pasted image 0 (1)](https://user-images.githubusercontent.com/78655282/196054004-6ae779f5-b85b-4828-b264-8a1f8175d7c5.png)


In Knet, we use both semantic and instance kernels to be applied on feature maps to give a group of masks that contain both semantic and instance masks, more likely as panoptic segmentation. Each kernel produces a mask, so combining these kernels is necessary to achieve this.


![pasted image 0 (3)](https://user-images.githubusercontent.com/78655282/196054072-f88efaf4-b83c-4e95-9578-9caab5aa7bc9.png)


This figure shows the architecture of Knet for segmentation purposes. The Kernel update Head is the main component that defines the dynamic behavior of kernels, which is different from the previous architectures. This differentiation arises because of their static kernels once trained and model weights saved. The Head takes feature maps, previous block-learned kernels, and mask predictions. After processing, it gives the newly learned kernel dynamically set that, in interaction with the previous mask predictions, gives new mask predictions but with iterative refinement. This process is repeated till the number of steps is defined after which final masks are predicted.


## How to train?

Considering the K-net GitHub repository, there are two types of dataset that you can use for training K-net (ADEChallengeData2016 and coco2017). If the labels of your dataset are not in these types, you can convert them by using well-known method to coco and ADE type. Also, if you don’t have any dataset, you can use famous prepared datasets like coco on the internet. You can download coco2017 dataset from [here](https://cocodataset.org/#download) using wget command.

In the second step, you need to a python environment. You can use python virtual environment, docker image or conda python environment. In my test I used python3.9 virtual environment and I recommend it to use. To guide you, you can read the documentation [here](https://docs.python.org/3/library/venv.html).

The hardest part of preparing the system to train the K-net is installing match dependencies. In this part, I totally suggest you following the exact visions in below for each library. 

The hardest part of preparing the system to train the K-net is installing match dependencies. In this part, I totally suggest you following the exact visions in below for each library. 

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install openmim==0.1.5 scipy mmdet==2.17.0 mmsegmentation==0.18.0
pip install git+https://github.com/cocodataset/panopticapi.git
mim install mmcv-full==1.3.14

```

The next step is to clone the K-net repository. You can find K-net repository from [here](https://github.com/ZwwWayne/K-Net). After cloning the repository, you have to make data directory in K-net folder like below :

```
data/
├── ade
│   ├── ADEChallengeData2016
│   │   ├── annotations
│   │   ├── images
├── coco
│   ├── annotations
│   │   ├── panoptic_{train,val}2017.json
│   │   ├── instance_{train,val}2017.json
│   │   ├── panoptic_{train,val}2017/  # panoptic png annotations
│   │   ├── image_info_test-dev2017.json  # for test-dev submissions
│   ├── train2017
│   ├── val2017
│   ├── test2017

```

In this step, you need to choose a config for training your dataset. You can find prepared configs for each dataset in K-net repository. In config file you can find dataset, model, and scheduler config for training and customize them for your dataset. For example, for Instance Segmentation on COCO, I have used knet_s3_r50_fpn_1x_coco. In this config file it can be seen ../_base_/datasets/coco_instance.py that shows the path of config dataset. You have gone there and change data_root in this file. Also, due to the type of your dataset, you have to  find suitable config in your python environment. For example , I made a python3.9 virtual environment and because I had used coco dataset type I changed coco.py file in path below : 
 environment/lib/python3.9/site-packages/mmdet/datasets/coco.py
You have to change class names in this file to consider to your dataset.

Finally, in knet_s3_r50_fpn_1x_coco needs to change num_classes option.  My dataset had 18 classes, so I changed this option from 80 to 18.


Finally, you can start training using command below : 

```
PYTHONPATH=‘.’:‘./K-Net’ mim train mmdet ./K-Net/configs/det/knet/knet_s3_r50_fpn_1x_coco.py --work-dir ./result

```
To test the model, you can use command below : 

```

PYTHONPATH=‘.’:‘K-Net/’ mim test mmdet K-Net/configs/det/knet/knet_s3_r50_fpn_1x_coco.py --checkpoint K-Net/results/epoch_12.pth --show-dir K-Net/results/ 

```

## Some outputs : 

![car4](https://user-images.githubusercontent.com/78655282/196054551-6b5f7a71-8bba-4d64-b14e-c3a7936bb8e2.jpg)


![car5](https://user-images.githubusercontent.com/78655282/196054564-087852f0-e59f-438a-b63e-98c8beb6d240.jpg)


![te95](https://user-images.githubusercontent.com/78655282/196054577-6df393b7-9769-428b-8f13-abdfd7fea7d6.jpg)


