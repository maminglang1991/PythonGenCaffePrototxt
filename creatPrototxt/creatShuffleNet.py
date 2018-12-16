# -*- coding: utf-8 -*-

#from caffe import layers as L,params as P,to_proto
from caffe import layers as L,params as P
import caffe
path= ''
train_list = '/home/caffe/work_dir/train_list.txt'
val_list = '/home/caffe/work_dir/test_list.txt'           
train_proto = path + 'train_landmark_shufflenet.prototxt'   
deploy_proto = path + 'deploy_landmark_shufflenet.prototxt'       

def after_conv(conv):
    bn = L.BatchNorm(conv, use_global_stats = False, in_place = False)
    scale = L.Scale(bn, bias_term = True, in_place = True)
    relu=L.ReLU(scale, in_place = True)
    return relu

def data_block(input, kernel_size, num_output, stride, pad):
    conv_data = L.Convolution(input, kernel_size = kernel_size, stride = 2, num_output = num_output, bias_term = False, pad = pad, weight_filler = dict(type = 'xavier'))
    conv_dw = L.Convolution(conv_data, kernel_size = kernel_size, stride = 1, num_output = num_output, group = num_output, bias_term = False, pad = pad, weight_filler = dict(type='xavier'))
    conv_dw_ = L.Convolution(conv_dw, kernel_size = 1, stride = 1,num_output = num_output, bias_term = False, pad = 0, weight_filler = dict(type = 'xavier'))
    data_eltwise = L.Eltwise(conv_data, conv_dw_, eltwise_param = dict(operation=1))
    return data_eltwise

def shuffle_block(input, kernel_size, num_middle, num_output, stride, pad):
    conv_up   = L.Convolution(input, kernel_size = 1, stride = 1,num_output = num_middle, bias_term = False, pad = 0, weight_filler = dict(type='xavier'))
    conv_dw   = L.Convolution(conv_up, kernel_size = kernel_size, stride = stride, num_output = num_middle, group = num_middle, bias_term = False, pad = pad, weight_filler = dict(type='xavier'))
    conv = L.Convolution(conv_dw, kernel_size = 1, stride = 1,num_output = num_output, bias_term = False, pad = 0, weight_filler = dict(type = 'xavier'))
    conv = after_conv(conv)
    shuffle_eltwise = L.Eltwise(input, conv, eltwise_param = dict(operation=1))
    return shuffle_eltwise

def mobile2_block(input, kernel_size, num_middle, num_output, stride, pad):
    conv_up  = L.Convolution(input, kernel_size = 1, stride = 1,num_output = num_middle, bias_term = False, pad = 0, weight_filler = dict(type='xavier'))
    conv_dw  = L.Convolution(conv_up, kernel_size = kernel_size, stride = stride, num_output = num_middle, group = num_middle, bias_term = False, pad = pad, weight_filler = dict(type='xavier'))
    conv_dw_ = L.Convolution(conv_dw, kernel_size = 1, stride = 1,num_output = num_output, bias_term = False, pad = 0, weight_filler = dict(type = 'xavier'))
    return conv_dw_




def create_landmark_net(train_list, batch_size):
    spec = caffe.NetSpec()
    spec.data, spec.label = L.ImageData(source=train_list, batch_size=batch_size, shuffle=True, ntop=2,
                           transform_param=dict(crop_size=112, mirror=False, scale=0.0078125, mean_value=127.5), include=dict(phase=caffe.TRAIN))
    spec.conv0  = data_block(spec.data,      kernel_size = 3, num_output = 8,  stride = 2, pad = 1)#采样
    spec.conv1  = mobile2_block(spec.conv0,  kernel_size = 3, num_middle = 48,  num_output = 8,  stride = 2, pad = 1)#采样
    spec.conv2  = shuffle_block(spec.conv1,  kernel_size = 3, num_middle = 48,  num_output = 8,  stride = 1, pad = 1)
    spec.conv3  = mobile2_block(spec.conv2,  kernel_size = 3, num_middle = 48,  num_output = 8,  stride = 2, pad = 1)#采样
    spec.conv4  = shuffle_block(spec.conv3,  kernel_size = 3, num_middle = 48,  num_output = 8,  stride = 1, pad = 1)
    spec.conv5  = shuffle_block(spec.conv4,  kernel_size = 3, num_middle = 48,  num_output = 8,  stride = 1, pad = 1)
    spec.conv6  = mobile2_block(spec.conv5,  kernel_size = 3, num_middle = 48,  num_output = 16, stride = 2, pad = 1)#采样
    spec.conv7  = shuffle_block(spec.conv6,  kernel_size = 3, num_middle = 96,  num_output = 16, stride = 1, pad = 1)
    spec.conv8  = shuffle_block(spec.conv7,  kernel_size = 3, num_middle = 96,  num_output = 16, stride = 1, pad = 1)
    spec.conv9  = shuffle_block(spec.conv8,  kernel_size = 3, num_middle = 96,  num_output = 16, stride = 1, pad = 1)
    spec.conv10 = mobile2_block(spec.conv9,  kernel_size = 3, num_middle = 96,  num_output = 24, stride = 1, pad = 1)
    spec.conv11 = shuffle_block(spec.conv10, kernel_size = 3, num_middle = 144, num_output = 24, stride = 1, pad = 1)
    spec.conv12 = shuffle_block(spec.conv11, kernel_size = 3, num_middle = 144, num_output = 24, stride = 1, pad = 1)
    spec.conv13 = mobile2_block(spec.conv12, kernel_size = 3, num_middle = 144, num_output = 40, stride = 2, pad = 1)#采样
    spec.conv14 = shuffle_block(spec.conv13, kernel_size = 3, num_middle = 240, num_output = 40, stride = 1, pad = 1)
    spec.conv15 = shuffle_block(spec.conv14, kernel_size = 3, num_middle = 240, num_output = 40, stride = 1, pad = 1)
    spec.conv16 = mobile2_block(spec.conv15, kernel_size = 3, num_middle = 240, num_output = 80, stride = 1, pad = 1)

    spec.conv17 = L.Convolution(spec.conv16, kernel_size = 1, stride = 1, num_output = 1280, bias_term = False, pad = 0, weight_filler = dict(type = 'xavier'))
    spec.conv18 = L.Convolution(spec.conv17, kernel_size = 1, stride = 1, num_output = 64,   bias_term = False, pad = 0, weight_filler = dict(type = 'xavier'))
    
    spec.fc0  = L.InnerProduct(spec.conv18, num_output = 212, weight_filler=dict(type='xavier'))

    spec.loss = L.EuclideanLoss(spec.fc0, spec.label, include=dict(phase=caffe.TRAIN))

    return spec.to_proto()


def write_net():
    #
    with open(train_proto, 'w') as f:
        f.write(str(create_landmark_net(train_list,batch_size=96)))

    #    
    with open(deploy_proto, 'w') as f:
        f.write(str(create_landmark_net(val_list,batch_size=30)))

if __name__ == '__main__':
    write_net()