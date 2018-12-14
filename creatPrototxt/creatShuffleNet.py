# -*- coding: utf-8 -*-

#from caffe import layers as L,params as P,to_proto
from caffe import layers as L,params as P
import caffe
path=''
train_list='/home/caffe/work_dir/train_list.txt'
val_list='/home/caffe/work_dir/test_list.txt'           
train_proto=path+'denseNet1.prototxt'   
deploy_proto=path+'deploy_denseNet1.prototxt'       

def net_block(input,kernel_size=3,num_output=32,stride=1,pad=1,MAX_POOL=False,BN=False,GROUP=False):
    conv = None
    if GROUP:
        conv=L.Convolution(input, kernel_size=kernel_size, stride=stride,num_output=num_output,group=num_output,engine=1,bias_term=False, pad=pad,weight_filler=dict(type='xavier'))
    else:
        conv=L.Convolution(input, kernel_size=kernel_size, stride=stride,num_output=num_output,bias_term=False, pad=pad,weight_filler=dict(type='xavier'))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    if MAX_POOL:
        maxpool1=L.Pooling(conv, pool=P.Pooling.MAX,stride=2,kernel_size=3)
        bn = L.BatchNorm(maxpool1, use_global_stats=False, in_place=True)
        #scale = L.Scale(bn,filler=1,bias_term=true,bias_filler=0)
        scale = L.Scale(bn,filler=1,bias_term=true,bias_filler=0)
        relu=L.ReLU(scale, in_place=True)
        return maxpool1,relu
    else:
        if BN:
            bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
            scale = L.Scale(bn,filler=1,bias_term=true,bias_filler=0)
            relu=L.ReLU(scale, in_place=True) 
        else:
            relu=L.ReLU(conv, in_place=True)     
    #relu=L.ReLU(bn, in_place=True)
        return relu

def shuffle_block(input, kernel_size, middle_output, num_output, stride, pad):
    conv = L.Convolution(input, kernel_size=1, stride=1,num_output=middle_output ,bias_term=False, pad=pad,weight_filler=dict(type='xavier'))

def eltwise_relu(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise_relu

def mobile_modeule(input, num_3x3,num_1x1,stride=1):
    #engine默认为0,1对应CAFFE,2对应CUDNN
    model3=L.Convolution(input, kernel_size=3, stride=stride,num_output=num_3x3,group=num_3x3,bias_term=False, pad=1,engine=1,weight_filler=dict(type='xavier'))
    model1=L.Convolution(model3, kernel_size=1, stride=1,num_output=num_1x1,bias_term=False, pad=0,weight_filler=dict(type='xavier'))
    bn = L.BatchNorm(model1, use_global_stats=False, in_place=True)
    relu=L.ReLU(bn,in_place=True)

    return relu   

def group_conv(input,num_output,GROUP=False,kernel_size=3,stride=1,pad=1):
    engine=0
    group=1
    if GROUP:
        engine=1
        group=num_output
    conv=L.Convolution(input, kernel_size=kernel_size, stride=stride,num_output=num_output,group=group,bias_term=False, pad=pad,engine=engine,weight_filler=dict(type='xavier'))
    return conv

def after_conv(conv):
    #in-place compute means your input and output has the same memory area,which will be more memory effienct
    bn = L.BatchNorm(conv, use_global_stats=False,in_place=False)
    #scale = L.Scale(bn,filler=dict(value=1),bias_filler=dict(value=0),bias_term=True, in_place=True)
    scale = L.Scale(bn,bias_term=True, in_place=True)
    relu=L.ReLU(scale, in_place=True)
    return relu

def res_block(input,stride=2,num_output=32,pad1=1,pad2=1,MAX_POOL=False):
    block1 = net_block(input=input,kernel_size=3,num_output=num_output,stride=stride,pad=pad1)
    block2 = net_block(input=block1,kernel_size=3,num_output=num_output,stride=1,pad=pad2)
    #block3 = net_block(input=block2,kernel_size=3,num_output=num_output,stride=1,pad=pad2)
    #block4 = eltwise_relu(block1,block2)
    residual_eltwise = L.Eltwise(block1, block2, eltwise_param=dict(operation=1))
    if MAX_POOL:
        maxpool1=L.Pooling(residual_eltwise, pool=P.Pooling.MAX,stride=2,kernel_size=3)
        bn = L.BatchNorm(maxpool1, use_global_stats=False, in_place=True)
        relu=L.ReLU(bn, in_place=True)
    else:
        bn = L.BatchNorm(residual_eltwise, use_global_stats=False, in_place=True)
        relu=L.ReLU(bn, in_place=True)        
    return relu

def concat_res_block(input):
    blockA1 = net_block(input=input,kernel_size=3,num_output=4,stride=2,pad=0)
    blockA2 = net_block(input=blockA1,kernel_size=3,num_output=4,stride=1,pad=0)
    blockA3 = eltwise_relu(blockA1,blockA2)

    blockB1 = net_block(input=input,kernel_size=3,num_output=4,stride=2,pad=0)
    blockB2 = net_block(input=blockB1,kernel_size=3,num_output=4,stride=1,pad=0)
    blockB3 = eltwise_relu(blockB1,blockB2)

    concatAB = L.Concat(blockA3, blockB3)    

'''def create_net(img_list,batch_size,include_acc=False):
    data,label=L.ImageData(source=img_list,batch_size=batch_size,shuffle=true,new_width=120,new_height=120,ntop=2,
                           transform_param=dict(crop_size=112,mirror=False,scale=0.0078125,mean_value=127.5))'''
def create_net(train_list,batch_size,include_acc=False):
    spec = caffe.NetSpec()   
    '''NetSpec可以用作命名,下面每一个spec.后面的字符直接就作为了该层的名字,没有使用spec的,系统会自动生成,在函数中的命名就是自动生成的,因为无法传递spec
 spec.data,spec.label=L.ImageData(source=train_list,batch_size=batch_size,shuffle=True,ntop=2,                       transform_param=dict(crop_size=112,mirror=False,scale=0.0078125,mean_value=127.5),phase=0) '''
    spec.data,spec.label=L.ImageData(source=train_list,batch_size=batch_size,shuffle=True,ntop=2,
                           transform_param=dict(crop_size=112,mirror=False,scale=0.0078125,mean_value=127.5),include=dict(phase=caffe.TRAIN))

    spec.conv1=group_conv(spec.data,kernel_size=3,num_output=32,stride=2)
    spec.relu1=after_conv(spec.conv1)
    spec.conv2=group_conv(spec.relu1,num_output=32,GROUP=True,kernel_size=3,stride=1)
    spec.relu2=after_conv(spec.conv2)
    spec.concat1=L.Concat(spec.relu1,spec.relu2,axis=1)
    spec.pooling1 = L.Pooling(spec.concat1,pool=P.Pooling.MAX,stride=2,kernel_size=3)
    spec.relu3=after_conv(spec.pooling1)

    spec.conv3=group_conv(spec.relu3,num_output=64,kernel_size=1,stride=1,pad=0)
    spec.relu4=after_conv(spec.conv3)
    spec.conv4=group_conv(spec.relu4,num_output=64,GROUP=True,kernel_size=3,stride=1)
    spec.relu5=after_conv(spec.conv4)

    spec.concat2=L.Concat(spec.pooling1,spec.relu5,axis=1)
    spec.relu6=after_conv(spec.concat2)
    spec.conv5=group_conv(spec.relu6,num_output=128,kernel_size=1,stride=1,pad=0)
    spec.relu7=after_conv(spec.conv5)
    spec.conv6=group_conv(spec.relu7,num_output=128,GROUP=True,kernel_size=3,stride=1)
    spec.relu8=after_conv(spec.conv6)
    spec.concat3=L.Concat(spec.concat2,spec.relu8,axis=1)

    spec.pooling2 = L.Pooling(spec.concat3,pool=P.Pooling.MAX,stride=2,kernel_size=3)
    spec.relu8=after_conv(spec.pooling2)

    spec.conv7=group_conv(spec.relu8,num_output=256,kernel_size=1,stride=1,pad=0)
    spec.relu9=after_conv(spec.conv7)
    spec.conv8=group_conv(spec.relu9,num_output=256,GROUP=True,kernel_size=3,stride=1)
    spec.relu10=after_conv(spec.conv8)    

    spec.concat4=L.Concat(spec.pooling2,spec.relu10,axis=1)
    spec.relu11=after_conv(spec.concat4)

    spec.conv9=group_conv(spec.relu11,num_output=512,kernel_size=1,stride=1,pad=0)
    spec.relu12=after_conv(spec.conv9)
    spec.conv10=group_conv(spec.relu12,num_output=512,GROUP=True,kernel_size=3,stride=1)
    spec.relu13=after_conv(spec.conv10)

    spec.concat5=L.Concat(spec.concat4,spec.relu13,axis=1)
    spec.relu14=after_conv(spec.concat5)
    spec.pooling3 = L.Pooling(spec.relu14,pool=P.Pooling.MAX,stride=2,kernel_size=3)

    spec.relu15=after_conv(spec.pooling3)
    spec.conv11=group_conv(spec.relu15,num_output=1024,kernel_size=1,stride=1,pad=0)
    spec.relu16=after_conv(spec.conv11)
    spec.conv12=group_conv(spec.relu16,num_output=1024,GROUP=True,kernel_size=3,stride=1)
    spec.relu17=after_conv(spec.conv12)

    #OUT 7
    spec.maxpool=L.Pooling(spec.relu17, pool=P.Pooling.AVE,global_pooling=True)
    spec.fc1=L.InnerProduct(spec.maxpool, num_output=1024,weight_filler=dict(type='xavier'))
    #relu1=L.ReLU(fc1, in_place=True)
    spec.fc2 = L.InnerProduct(spec.fc1, num_output=10000,weight_filler=dict(type='xavier'))
    #,phase=0,0对应TRAIN
    spec.loss = L.SoftmaxWithLoss(spec.fc2, spec.label,include=dict(phase=caffe.TRAIN))

    #acc = L.Accuracy(fc2, label)
    #return to_proto(loss, acc,include=dict(phase=TEST))

    if include_acc:             
        return caffe.to_proto(spec.fc1)
    else:
        spec.acc = L.Accuracy(spec.fc2, spec.label,include=dict(phase=caffe.TEST))
        #return spec.to_proto(spec.loss, spec.acc)
        return spec.to_proto()


def write_net():
    #
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_list,batch_size=96)))

    #    
    with open(deploy_proto, 'w') as f:
        f.write(str(create_net(val_list,batch_size=30, include_acc=True)))

if __name__ == '__main__':
    write_net()