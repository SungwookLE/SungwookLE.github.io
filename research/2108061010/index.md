---
layout: post
type: research
date: 2021-08-06 10:10
category: Pytorch
title: AI- Pytorch Studying
subtitle: Tutorial- Pytorch quick start
post-header: true
header-img: 
hash-tag: [Pytorch]
use_math: true
---

# Pytorch: Quick Start
AUTHOR: SungwookLE  
DATE: '21.8/6  
TUTORIAL: [Tutorial Pytorch Korea](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)  

## 빠른 시작(QuickStart)

- [link](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)
이번 장에서는 기계 학습의 일반적인 작업들을 위한 API를 통해 실행됩니다. 더 자세히 알아보려면 각 장(section)의 링크를 참고하세요.


```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

```


```python
torch.cuda.torch.cuda.is_available()
```




    True



## 1. 데이터 작업하기
파이토치(PyTorch)에는 데이터 작업을 위한 기본 요소 두가지인 torch.utils.data.DataLoader 와 torch.utils.data.Dataset 가 있습니다. Dataset 은 샘플과 정답(label)을 저장하고, DataLoader 는 Dataset 을 반복 가능한 객체(iterable)로 감쌉니다.


```python
# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz


    26422272it [00:11, 2290858.89it/s]                              


    Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz


    29696it [00:00, 123312.91it/s]                          


    Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz


    4422656it [00:08, 528212.22it/s]                              


    Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz


    6144it [00:00, 32292987.19it/s]         

    Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    


    
    /home/joker1251/anaconda3/envs/36py_wooks/lib/python3.6/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


- PyTorch는 TorchText, TorchVision 및 TorchAudio 와 같이 도메인 특화 라이브러리를 데이터셋과 함께 제공하고 있습니다. 이 튜토리얼에서는 TorchVision 데이터셋을 사용하도록 하겠습니다.

- torchvision.datasets 모듈은 CIFAR, COCO 등과 같은 다양한 실제 비전(vision) 데이터에 대한 Dataset(전체 목록은 여기)을 포함하고 있습니다. 이 튜토리얼에서는 FasionMNIST 데이터셋을 사용합니다. 모든 TorchVision Dataset 은 샘플과 정답을 각각 변경하기 위한 transform 과 target_transform 의 두 인자를 포함합니다.


```python
# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

- Dataset 을 DataLoader 의 인자로 전달합니다. 이는 데이터셋을 반복 가능한 객체(iterable)로 감싸고, 자동화된 배치(batch), 샘플링(sampling), 섞기(shuffle) 및 다증 프로세스로 데이터 불러오기(multiprocess data loading)를 지원합니다. 여기서는 배치 크기(batch size)를 64로 정의합니다. 즉, 데이터로더(dataloader) 객체의 각 요소는 64개의 특징(feature)과 정답(label)을 묶음(batch)으로 반환합니다.


```python
batch_size = 64

# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)
```


```python
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
```

    Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
    Shape of y:  torch.Size([64]) torch.int64


## 2. 모델 만들기(*)

- Pytorch에서 신경망 모델은 nn.Module을 상속받는 클래스(class)를 생성하여 정의합니다. __init__ 함수에서 신경망의 계층(layer)들을 정의하고 `forward`함수에서 데이터를 어떻게 전달할지 지정합니다. 가능한 경우 GPU로 신경망을 이동시켜 연산을 가속(accelerate)합니다.


```python
# 학습에 사용할 CPU나 GPU 장치를 얻습니다.
device = "cuda" if torch.cuda.torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
```

    Using cuda device



```python
# 모델을 정의합니다.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
        (5): ReLU()
      )
    )


## 3. 모델 매개변수 최적화하기
- 모델을 학습하려면 손실 함수(loss function) 와 옵티마이저(optimizer) 가 필요합니다.


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

- 각 학습 단계(training loop)에서 모델은 (배치(batch)로 제공되는) 학습 데이터셋에 대한 예측을 수행하고, 예측 오류를 역전파하여 모델의 매개변수를 조정합니다.


```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

- 모델이 학습하고 있는지를 확인하기 위해 테스트 데이터셋으로 모델의 성능을 확인합니다.


```python
def test(dataloader, model, loss_fn):
    size  = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct): >0.1f}%, Avg loss: {test_loss:>8f} \n")
```

- 학습 단계는 여러번의 반복 단계(epochs)를 거쳐서 수행됩니다. 각 에폭에서는 모델은 더 나은 예측을 하기 위해 매개변수를 학습한니다. 각 에폭마다 모델의 정확도(accuracy)와 손실(loss)을 출력합니다. 에폭마다 정확도가 증가하고 손실이 감소하는 것을 보려고 합니다.


```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1} \n-----------------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

    Epoch 1 
    -----------------------------------------------
    loss: 2.193637  [    0/60000]
    loss: 2.203587  [ 6400/60000]
    loss: 2.175505  [12800/60000]
    loss: 2.211402  [19200/60000]
    loss: 2.133227  [25600/60000]
    loss: 2.079954  [32000/60000]
    loss: 2.122299  [38400/60000]
    loss: 2.062492  [44800/60000]
    loss: 2.065115  [51200/60000]
    loss: 2.023016  [57600/60000]
    Test Error: 
     Accuracy: 41.6%, Avg loss: 2.057532 
    
    Epoch 2 
    -----------------------------------------------
    loss: 2.016894  [    0/60000]
    loss: 2.046291  [ 6400/60000]
    loss: 1.994132  [12800/60000]
    loss: 2.094479  [19200/60000]
    loss: 1.935474  [25600/60000]
    loss: 1.845785  [32000/60000]
    loss: 1.939914  [38400/60000]
    loss: 1.831414  [44800/60000]
    loss: 1.853358  [51200/60000]
    loss: 1.802729  [57600/60000]
    Test Error: 
     Accuracy: 44.8%, Avg loss: 1.867658 
    
    Epoch 3 
    -----------------------------------------------
    loss: 1.781377  [    0/60000]
    loss: 1.849651  [ 6400/60000]
    loss: 1.779787  [12800/60000]
    loss: 1.958196  [19200/60000]
    loss: 1.730441  [25600/60000]
    loss: 1.622214  [32000/60000]
    loss: 1.758102  [38400/60000]
    loss: 1.624734  [44800/60000]
    loss: 1.650914  [51200/60000]
    loss: 1.613259  [57600/60000]
    Test Error: 
     Accuracy: 46.3%, Avg loss: 1.702955 
    
    Epoch 4 
    -----------------------------------------------
    loss: 1.567205  [    0/60000]
    loss: 1.685313  [ 6400/60000]
    loss: 1.603254  [12800/60000]
    loss: 1.845334  [19200/60000]
    loss: 1.576494  [25600/60000]
    loss: 1.461073  [32000/60000]
    loss: 1.619784  [38400/60000]
    loss: 1.477227  [44800/60000]
    loss: 1.504015  [51200/60000]
    loss: 1.482282  [57600/60000]
    Test Error: 
     Accuracy: 47.0%, Avg loss: 1.581692 
    
    Epoch 5 
    -----------------------------------------------
    loss: 1.411528  [    0/60000]
    loss: 1.563468  [ 6400/60000]
    loss: 1.470545  [12800/60000]
    loss: 1.759837  [19200/60000]
    loss: 1.468434  [25600/60000]
    loss: 1.350994  [32000/60000]
    loss: 1.525407  [38400/60000]
    loss: 1.375344  [44800/60000]
    loss: 1.406244  [51200/60000]
    loss: 1.393930  [57600/60000]
    Test Error: 
     Accuracy: 47.9%, Avg loss: 1.492919 
    
    Done!




## 4. 모델 저장하기
- 모델을 저장하는 일반적인 방법은 (모델의 매개변수들을 포함하여) 내부 상태 사전(internal state dictionary)을 직렬화(serialize)하는 것입니다.


```python
torch.save(model.state_dict(), "model.pth")
print("Saved Pytorch Model State to model.pth")
```

    Saved Pytorch Model State to model.pth


## 5. 모델 불러오기
- 모델을 불러오는 과정에는 모델 구조를 다시 만들고 상태 사전을 모델에 불러오는 과정이 포함됩니다.


```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```




    <All keys matched successfully>




```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
```


```python
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

    Predicted: "Ankle boot", Actual: "Ankle boot"

## 끝
