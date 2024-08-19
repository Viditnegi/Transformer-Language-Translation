# Broadcasting
When broadcasting tensors in PyTorch, the broadcasting rules expand the smaller tensor to match the larger tensor's shape without actually copying the data. The broadcasting rules are as follows:

Starting from the trailing dimensions (rightmost), PyTorch compares the dimensions of the two tensors.
Two dimensions are compatible if:
They are equal.
One of them is 1.
One of them does not exist (i.e., a smaller tensor can be expanded by adding a new dimension of size 1).
If the dimensions are compatible, PyTorch can broadcast the smaller tensor to match the larger tensor.


