# A Simple Implementation of DenseNet in Pytorch

This project is a homework of course COMP9444, UNSW. The main.py file and dataset are given by [Alan Blair](https://www.cse.unsw.edu.au/~blair/). The model I selected is DenseNet-169.

DenseNet is one of the popular CNNs on image classification tasks. This project is a private practice on building a DenseNet following the paper:

[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

I test my model on the supplementary  

Furthermore, the following papers also help me on training model and selecting hyperparameters:

1. [Train longer, generalize better: closing the generalization gap in large batch training of neural networks](https://arxiv.org/abs/1705.08741)
2. [Data augmentation instead of explicit regularization](https://arxiv.org/abs/1806.03852)
3. [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)

# Training details

The architecture of my model is referred to DenseNet-169. There are four dense blocks and size of each is 6,12,24,16. The grow rate is 24. I apply heavy transformations including horizontal flip, rotation(from -45 to 45 degree), color jitter and random crop. I normalize the input by mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] which are common mean-std values from ImageNet.

The optimizer selected is SGD with 0.9 Nesterov momentum and 1e-4 weight decay. The learning rate is 0.01, although changing learning rate at different proportions of total epochs might obtain a better performance. I do not provide Dropout in my model since the dataset is not large. This idea comes from the research result of [Data augmentation instead of explicit regularization](https://arxiv.org/abs/1806.03852), data augmentation is better than regularizations especially on smaller dataset. I would like to reappear this in the further works.

For time efficiency, the total epochs are 200 (300 in the original paper) and the batch size is 64. Some researchers demonstrate that too large of batch size will lead to poor generalization. The size of validation set is 0.2 of the whole training set.

# Description of Dataset

This dataset is given by my lecturer [Alan Blair](https://www.cse.unsw.edu.au/~blair/). But I believe there is an original source for this dataset online. The task is to classify the different characters in cartoon Simpson Family. There are 14 characters in the dataset. The size of input image is 64 by 64. The number of channel is 3 although the images look like grayscale images.

# Result

The accuracy on validation set could reach around 96%. The accuracy test on selected 700 test samples is 97.14%.
