# PyTorch-Notes
Listing my PyTorch Notes

1) Tensors are core data abstractions of PyTorch. Our inputs, outputs and weights are all tensors.

2) Autograd is pytorch's automatic differentation engine. The backward pass of our model is done with a single function call.

## Tensors

3) In numpy, there are vectors and arrays. However, everything in pytorch is a tensor. Tensor can be 1-dimensional, 2-dimensional, 3-dimensional or more.

4) To create an empty tensor in different dimensions

```run.py
import torch

x_scalar = torch.empty(1)
x_vector = torch.empty(3)
x_matrix = torch.empty(2,3)
x_more_dimensions = torch.empty(4,5,6)

print(x_scalar.shape,x_vector.shape,x_matrix.shape,x_more_dimensions.shape)
#torch.Size([1]) torch.Size([3]) torch.Size([2, 3]) torch.Size([4, 5, 6])
```

5) To use numpy alike zeros, ones and rand in torch, run the following. We can make operations like mathematical operations like (+, -, *, /) in element wise.

```numpy_alike.py
import torch

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

```
import torch

x = torch.tensor([3,4,5])
print(x[0].item())# 3
```







