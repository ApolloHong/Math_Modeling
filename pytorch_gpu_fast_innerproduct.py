import torch

def inner_product(array1 , array2):
    array1 = torch.from_numpy(array1)
    array2 = torch.from_numpy(array2)

    # CPU计算
    prod = torch.dot(array1, array2)

    # GPU计算
    array1 = array1.cuda().float()
    array2 = array2.cuda().float()
    prod = torch.dot(array1, array2)