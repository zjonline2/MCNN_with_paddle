import paddle.v2 as paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import numpy as np
def convseg(input,num_filters,filter_size,max_pool):
    input=fluid.layers.conv2d(input=input,
                              num_filters=num_filters,
                              filter_size=filter_size,
                              stride=1,
                              padding=filter_size/2,
                              act='relu',
                              dilation=1,
                              param_attr=fluid.param_attr.ParamAttr(
                                 initializer=fluid.initializer.Normal(scale=0.01)),
                              bias_attr=fluid.param_attr.ParamAttr(
                                 initializer=fluid.initializer.Constant(value=0.0)))
    if max_pool:
       input=fluid.layers.pool2d(input=input,pool_size=2,pool_type='max')
    return input
def core(input):
    num_filter=[[[9,16,False],[7,32,True],[7,16,True],[7,8,False]],
                [[7,20,False],[5,40,True],[5,20,True],[5,10,False]],
                [[5,24,False],[3,48,True],[3,24,True],[3,12,False]]]
    convs=[];conv=input
    for i in num_filter:
        for j in i:
            conv=convseg(conv,j[1],j[0],j[2])
        convs.append(conv)
    return convs
def MCNN(input,pretrain):
    out=core(input)
    print out[0].shape;print out[1].shape;print out[2].shape

