# Vehicle-Make-and-Model-Recognition (VMMR)

## Data
Data used here is freely available from the following sources:
-[VMMRdb](https://www.dropbox.com/s/uwa7c5uz7cac7cw/VMMRdb.zip?dl=0)
-[Stanford Cars (Training)](http://ai.stanford.edu/~jkrause/car196/cars_train.tgz)
-[Stanford Cars (Test)](http://ai.stanford.edu/~jkrause/car196/cars_test.tgz)
-[Stanford Cars (Labels and Bounding Boxes)] (http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)

## Dependencies

## Installation
Download all the above described data sources and place them each in its appropriate folder. For example, the downloaded
training images folder `car_train` be placed in `StanfordCars`.

## Description
This task is an example of **Fine-Grained image Classification**.

Two popular VMMR datasets, Stanford Cars and VMMRdb, were unified for this undertaking. This was to ensure adequate 
amounts of training data. Especially from the VMMRdb data, images are of varying qualities and taken from multiple
view angles; hopefully this will allow for greater generalization ability of the trained network.

## References
-[VMMRdb](https://github.com/faezetta/VMMRdb)
- Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. "A Large-Scale Car Dataset for Fine-Grained Categorization
and Verification", In Computer Vision and Pattern Recognition (CVPR), 2015.
- 