import torch
import torch.nn as nn
from helper_funcs import conv_block, res_block


width = 36
height = 36
stride = 1
padding = 1
filter_size = 6
number_of_filters = 32
input_channels = 3 #RGB

x = torch.zeros([1,512,width,height])

y = torch.zeros([1,256,width,height])

res1 = nn.Sequential(res_block(512, 3, padding = 1), res_block(512, 1, padding = 0))
conv1 = conv_block(256, 512, False, filter_size=3, padding = 0)
y = conv1(y)
x = res1(x)

print(f"Output width formular = {width} - {filter_size} + 2* {padding} / {stride} + 1 = {int(((width-filter_size + 2 * padding)/stride) + 1 )}")
print(f"Output height formular = {height} - {filter_size} + 2* {padding} / {stride} + 1 =", int((height-filter_size + 2 * padding)/stride + 1 ))

print("Resnet output: ", x.shape)
print("conv output: ", y.shape)
