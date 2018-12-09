# Download dataset
* Download mnist from [here](http://yann.lecun.com/exdb/mnist/).
* Download cifar10(binary version) or cifar100(binary version) from [here](http://www.cs.toronto.edu/~kriz/cifar.html).
* Download svhn(Cropped Digits) from [here](http://ufldl.stanford.edu/housenumbers/).

# mnist
mnist directory should contain following files:
* train-images-idx3-ubyte.gz
* train-labels-idx1-ubyte.gz
* t10k-images-idx3-ubyte.gz
* t10k-labels-idx1-ubyte.gz

# cifar10
cifar10 directory should as least contain following files:
* data_batch_[1-5].bin
* test_batch.bin

# cifar100
cifar100 directory should as least contain following files:
* train.bin
* test.bin

# svhn
svhn directory should as least contain following files:
* train_32x32.mat
* test_32x32.mat