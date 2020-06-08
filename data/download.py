import torchvision.datasets as tvd

if __name__ == "__main__":
    data_dir = "./data"

    for train in [True, False]:
        tvd.MNIST(data_dir, download=True, train=train)
        tvd.CIFAR10(data_dir, download=True, train=train)
        tvd.Omniglot(data_dir, download=True, background=train)
