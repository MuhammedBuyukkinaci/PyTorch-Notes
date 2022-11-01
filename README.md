# PyTorch-Notes
Listing my PyTorch Notes

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

13) requires_grad is a parameter of tensor. It means the tensor is going to be optimized in training.

```
a = torch.rand(100,100,requires_grad =True)
```

## Autograd

14) We can calculate gradients using autograd.

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


