# Vehicle-Make-and-Model-Recognition (VMMR)

## Data
Data used here is freely available from the following sources:
-[VMMRdb](https://www.dropbox.com/s/uwa7c5uz7cac7cw/VMMRdb.zip?dl=0)
-[Stanford Cars (Training)](http://ai.stanford.edu/~jkrause/car196/cars_train.tgz)
-[Stanford Cars (Test)](http://ai.stanford.edu/~jkrause/car196/cars_test.tgz)
-[Stanford Cars (Labels and Bounding Boxes)] (http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)

## Dependencies
- Numpy
- Pandas
- Scikit-Image
- Torch
- Torchvision
- tqdm

## Installation
Download all the above described data sources and place them each in its appropriate folder. For example, the downloaded
training images folder `car_train` be placed in `StanfordCars`.

## Description
This task is an example of **Fine-Grained Image Classification**, and **Transfer Learning**. The initial hypothesis is to 
avoid fine-tuning; because we have limited data per class, we would be likely to overfit if we fine-tuned the entire network.
Instead, we will freeze all the network weights of the pretrained ConvNet, replace the final fully-connected layer to one with appropriate 
output dimension (random weight initialization); and treat the ConvNet as a fixed feature extractor. We work under the assumption that the ImageNet
pretrained features are similar enough to our vehicle dataset, and therefore we can train on top of the network (as
opposed to at a stage earlier in the network, where learned features are more general).

Two popular VMMR datasets, Stanford Cars and VMMRdb, were unified for this undertaking. This was to ensure adequate 
amounts of training data. Especially from the VMMRdb data, images are of varying qualities and taken from multiple
view angles; hopefully this will allow for greater generalization ability of the trained network.

The VMMRdb dataset had many classes with few datapoints. These classes and associated samples were removed.

Stratified sampling?

## References
-[VMMRdb](https://github.com/faezetta/VMMRdb)
- Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. "A Large-Scale Car Dataset for Fine-Grained Categorization
and Verification", In Computer Vision and Pattern Recognition (CVPR), 2015.
- [CS231n Transfer Learning Notes](https://cs231n.github.io/transfer-learning/)
- [Pytorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)