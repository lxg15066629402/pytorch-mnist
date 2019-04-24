from torchvision.datasets import mnist as mn
train_set = mn.MNIST('data', train=True, download=True)
test_set = mn.MNIST('data', train=False, download=True)  # test_pytorch/datatest_pytorch/data
