# PyTorch-Notes

Listing my PyTorch Notes. The source is [Deep Learning With PyTorch - Full Course](https://www.youtube.com/watch?v=c36lUUr864M)

1) Tensors are core data abstractions of PyTorch. Our inputs, outputs and weights are all tensors.

2) Autograd is pytorch's automatic differentation engine. The backward pass of our model is done with a single function call.

## Tensors

3) In numpy, there are vectors and arrays. However, everything in pytorch is a tensor. Tensor can be 1-dimensional, 2-dimensional, 3-dimensional or more.

4) To create an empty tensor in different dimensions

```run.py
x_scalar = torch.empty(1)
x_vector = torch.empty(3)
x_matrix = torch.empty(2,3)
x_more_dimensions = torch.empty(4,5,6)

print(x_scalar.shape,x_vector.shape,x_matrix.shape,x_more_dimensions.shape)
#torch.Size([1]) torch.Size([3]) torch.Size([2, 3]) torch.Size([4, 5, 6])
```

5) To use numpy alike zeros, ones and rand in torch, run the following. We can make operations like mathematical operations like (+, -, *, /) in element wise.

```numpy_alike.py
x_ones = torch.ones(3,4, dtype=torch.float16)
x_random = torch.rand(3,4,dtype=torch.double)

x_via_operator = x_ones + x_random
x_via_method = torch.add(x_ones,x_random)

x_zeros = torch.zeros(1,3,4)
```

6) To create a torch tensorf from a python list, x = torch.tensor([3,4,5])

7) In pytorch, every function that has a trailing underscore(\_) is making inplace operation. Examples are torch.add\_, torch.sub\_, torch.mul\_, torch.div\_.

8) We can use colon(:) in sciling operations as we did in numpy.

9) If there is only one value in our tensor, we can access it via **.item()**.

```item_example.py
x = torch.tensor([3,4,5])
print(x[0].item())# 3
```

10) To reshape a tensor, use view **torch.view()**

```reshape.py

x = torch.rand(8,8)
y = x.view(-1,16)
print(y.size())#torch.Size([4, 16])
z = x.view(32,2)
print(z.size())#torch.Size([32, 2])
```

11) If we want to convert a torch array to numpy array, use **.numpy()** function. If we are using pytorch on cpu and converting a pytorch tensorf to a numpy array, they are using  the same memory location. Thus, an update to a torch tensor will alter the numpy array or vice versa(an update to a numpy array will calter the torch tensor). A tensor moved to GPU can't be converted to numpy array.

```numpy_convert.py
import numpy as np
import torch

a = torch.rand(3,4)
a_numpy = a.numpy()
print(type(a), type(a_numpy))#<class 'torch.Tensor'> <class 'numpy.ndarray'>

b = np.ones(4)
b_torch = torch.from_numpy(b)
print(type(b), type(b_torch))


b = np.ones(4)
b_torch = torch.from_numpy(b)
print(type(b), type(b_torch))#<class 'numpy.ndarray'> <class 'torch.Tensor'>

```

12) To move a cpu tensor to GPU or vice versa, use `.to()` method of tensor.

```cuda_usage.py
if torch.cuda.is_available():
    device = torch.device("cuda")
    #first way to move a tensor to GPU
    x = torch.ones(4, device=device)
    # second way to move a tensor to GPU
    y = torch.rand(4)
    y = y.to(device=device)
    #z is a tensor on GPU
    z = x + y
    #can't convert torch gpu tensor to numpy, therefor convert it to cpu tensor first and then convert to numpy array
    z = z.to("cpu")
    z_numpy = z.numpy()
    print(type(z),type(z_numpy))#<class 'torch.Tensor'> <class 'numpy.ndarray'>

```

## Autograd

13) requires_grad is a parameter of tensor. It means the tensor is going to be optimized in training. In order for PyTorch to disable requires_grad parameter after creation, use the 2nd code snippet.

```run.py
a = torch.rand(100,100,requires_grad =True)
```

```disable.py
x = torch.randn(3,requires_grad=True)
#way 1
x.requires_grad_(False)
# way 2
x.detach()
# way 3
with torch.no_grad():
    y = x + 2
    # printing y and requires_grad is False
    print(y)
```

14) We can calculate gradients using autograd. In order to apply backward operation, the last elemnt should be scalar.

```autograd_usage.py
import numpy as np
import torch

a = torch.rand(3,requires_grad =True)
b = a + 2
c = b*b*4
d = c.mean()
print(d)
d.backward()#tensor(20.5779, grad_fn=<MeanBackward0>)
print(a.grad)#tensor([6.0689, 5.9862, 6.0895])

```

15) While looping in PyTorch, always set gradients of weights to 0.

```
weights = torch.ones(4,requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()

```

## Backpropagation

16) 3 steps of backpropagation

![](./images/001.png)

![](./images/002.png)

![](./images/003.png)

```
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0,requires_grad=True)

#forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)
#backward loss
loss.backward()
print(w.grad)
#update weights
#next forward and backwards

```

## Gradient Descent

17) A numpy implementation of Linear Regression

```numpy_example.py
import numpy as np

X= np.array([1,2,3,4], dtype=np.float32)
Y= np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w*x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x -y)**2
# dJ/dw = 1/N * 2 * x * (w*x -y)

def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted -y).mean()

print(f"Prediction before training: f(5) = {forward(5): .3f} ")

learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y,y_pred)
    # gradients
    dw = gradient(X,Y,y_pred)
    # update weights
    w -= learning_rate * dw

    print(f"epoch {epoch}, w = {w}, loss = {l}")

print(f"Prediction after training f(5) = {forward(5)}")
```

18) Pytorch implementation of Lİnear Regression

```torch_implementation.py
import torch

X= torch.tensor([1,2,3,4], dtype=torch.float32)
Y= torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# model prediction
def forward(x):
    return w*x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x -y)**2
# dJ/dw = 1/N * 2 * x * (w*x -y)

print(f"Prediction before training: f(5) = {forward(5): .3f} ")

learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y,y_pred)
    # gradients
    l.backward()
    
    # updateweights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # setting gradients zero
    w.grad.zero_()

    print(f"epoch {epoch}, w = {w}, loss = {l}")

print(f"Prediction after training f(5) = {forward(5)}")

```

## Training Pipeline

19) General training pipeline in PyTorch is composed of 3 steps.

    1) Design model(input, output, forward pass)
    2) Construct loss and optimizer
    3) Training loop
        - Forward pass: Compute Prediction
        - Backward pass: gradients
        - update weights

20) An example pytorch training pipeline

```training_pipeline.py
import torch
import torch.nn as nn

X  = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y  = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_dim=input_size, output_dim= output_size)


print(f"Prediction before training: f(5) = {model(X_test).item(): .3f} ")

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    # loss
    l = loss(Y,y_pred)
    # gradients
    l.backward()
    
    # updating weights
    optimizer.step()

    # setting gradients to zero
    optimizer.zero_grad()

    [w,b] = model.parameters()

    print(f"epoch {epoch}, w = {w[0][0].item()}, loss = {l}")

print(f"Prediction after training f(5) = {model(X_test).item(): .3f} ")

```

# Linear Regression

21) A linear regression implementation

```linear_reg.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets

# Step0: Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples, n_features = X.shape

#1) Step1: Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size,output_size)

# 2) Loss and optimizer
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# 3) Training Loop

num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted,y)

    #backward pass(back propogation and calculates gradients for us)
    loss.backward()

    # update(updating model weights)
    optimizer.step()

    # set gradients to 0
    optimizer.zero_grad()

    print(f"epoch: { epoch + 1}, loss = {loss.item()}")

# plot
# detach will create a new tensor, where our gradient calculation attribute is set to False
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted,'b')
plt.show()

```

## Logistic Regression

22) Logistic regression implementation

```logistic.py
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step0: Prepare data
bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target


n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 51)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#1) Step1: Model

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features) -> None:
        super().__init__()
        self.linear = nn.Linear(n_input_features,1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_input_features=n_features)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# 3) Training Loop

num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted,y_train)

    #backward pass(back propogation and calculates gradients for us)
    loss.backward()

    # update(updating model weights)
    optimizer.step()

    # set gradients to 0
    optimizer.zero_grad()

    print(f"epoch: { epoch + 1}, loss = {loss.item()}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")

```

## Dataset and DataLoader

23) If we use Dataset and DataLoader from pytorch, it will automatically deal with batch looping.

```dataset_dataloader.py
import math

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):

    def __init__(self) -> None:
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv',delimiter = ',', dtype = np.float32, skiprows = 1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])# n_samples, 1
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y_index

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size= 4, shuffle=True, num_workers=2)
# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations =  (total_samples/ 4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs,labels) in enumerate(dataloader):
        print(f"epoch: {epoch+1}/{num_epochs}, step {i + 1}/ {n_iterations}, inputs {inputs.shape}")
    
# built-in Pytorch dataset, cifar, coco, MNIST
# torchvision.datasets.MNIST()
```

## Dataset Transforms

24) Transform options are listed [here](https://pytorch.org/vision/stable/transforms.html). `torchvision.transforms.ToTensor()` is used a lot.

```transform_example.py
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class WineDataset(Dataset):

    def __init__(self, transform = None) -> None:
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv',delimiter = ',', dtype = np.float32, skiprows = 1)
        self.x = xy[:,1:]
        self.y = xy[:,[0]]# n_samples, 1
        self.n_samples = xy.shape[0]

        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.x[index], self.y_index
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self,factor) -> None:
        self.factor = factor
    
    def __call__(self, sample):
        inputs,target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=None)
dataset = WineDataset(transform=ToTensor())

composed = torchvision.transforms.Compose(
    [ToTensor(),MulTransform(2)]
)
dataset = WineDataset(transform=composed)

```

## Softmax and CrossEntropy

25) A basic usage of `torch.softmax`

```
x = torch.tensor([2,1,0.1])
x_logit = torch.softmax(x, dim=0)
print(x_logit)#tensor([0.6590, 0.2424, 0.0986])
```

26) Softmax function is usually combined with cross entropy loss.

![](./images/004.png)

27) nn.CrossEntropyLoss applies nn.LogSoftmax + nn.NLLLoss(negative log likelihood loss). **Don't put softmax in last layer**.
**The target(y) has class labels and it isn't one-hot encoded**. Y_pred has raw scores(logits), therefore softmax should't be implemented explicitly.

```loss.py
import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

y = torch.tensor([0])
#nsamples x nclasses = 1x3
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.1, 3.0, 0.5]])

l1 = loss(y_pred_good,y)
l2 = loss(y_pred_bad,y)
print(l1.item())#0.4170
print(l2.item())#3.0284

_, predictions1 = torch.max(y_pred_good,1)
_, predictions2 = torch.max(y_pred_bad,1)
print(predictions1)#tensor([0])
print(predictions2)#tensor([1])

y_multiple = torch.tensor([0,1,2,2,1])
y_pred_multiple = torch.tensor([
    [0.6,0.2,0.33],
    [0.4,0.1,0.1],
    [0.5,0.9,0.8],
    [0.1,1.99,3.4],
    [0.2,0.22,2.1]
])
l_multiple = loss(y_pred_multiple,y_multiple)
print(l_multiple)#tensor(1.1072)
```

28) An example NeuralNet for Multiclass classification

```neuralnet.py
import torch
import torch.nn as nn


class NeuralNetMulticlass(nn.Module):
    def __init__(self,input_size, hidden_size, num_classses) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.linear2= nn.Linear(hidden_size,num_classses)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #no softmax at the end
        return out

model = NeuralNetMulticlass(input_size=28*28, hidden_size=5, num_classses=3)
criterion = nn.CrossEntropyLoss()
```

29) Binary Classification example. EXPLICITLY define torch.sigmoid in forward method.

```binary.py
import torch
import torch.nn as nn


class NeuralNetBinary(nn.Module):
    def __init__(self,input_size, hidden_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.linear2= nn.Linear(hidden_size,1)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNetBinary(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()
```


