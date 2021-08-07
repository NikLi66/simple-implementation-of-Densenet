# A Simple Implementation of DenseNet in Pytorch

DenseNet is one of the popular CNNs on image classification tasks. This project is a private practice on building a DenseNet following the paper:

[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

I test my model on CIFAR-10, which is one of the most famous image dataset.

Furthermore, the following papers also help me on training model and selecting hyperparameters:

1. [Train longer, generalize better: closing the generalization gap in large batch training of neural networks](https://arxiv.org/abs/1705.08741)
2. [Data augmentation instead of explicit regularization](https://arxiv.org/abs/1806.03852)
3. [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)

# Training details

The architecture of my model consists of 3 blocks, size of each is 16(i.e L=100). I do not apply heavy transformations excepts horizontal flip. I normalize the input by mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010) which are common mean-std values used in CIFAR-10.

The optimizer selected is SGD with 0.9 Nesterov momentum and 1e-4 weight decay. The initial learning rate is 0.1 and divided by 10 at 50% and 75% of total epochs, respectively. Dropout is provided after each convolutional layer except the first one. However, the idea comes from [Data augmentation instead of explicit regularization](https://arxiv.org/abs/1806.03852) demonstrates that data augmentation is better than regularizations especially on smaller dataset. I would like to reappear this in the further works.

The total epochs are 300 (300 in the original paper) and the batch size is 64. Some researchers demonstrate that too large of batch size will lead to poor generalization.

All of my selections of hyperparameters are following the implementation details from [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993). This project aims to reappear the similar result.

# Environment and Result

My model is excuted on CUDA. The version of torch is 1.8.1 and torchvision is 0.9.1.

The accuracy on training set could finally reach around 96%. The accuracy test on selected 700 test samples is 97.14%.

# Discussion
My implemention still have many issues. The input channel of linear layer(i.e classifier) could not be calculated automatically. This requires setting the size of input image as a parameter of my model and calculate the input channel of classifier in terms of the number of dense blocks. Furthermore, the output channel of the first conv layer is default by double grow rate. This should be set as a parameter of DenseNet model.
