import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def max_pool(bottom, ks=3, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def ave_pool(bottom, ks=7, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)

def conv_bn_scale(bottom, nout, ks, stride, pad):
	conv = L.Convolution(bottom, kernel_size = ks, stride = stride, num_output = nout, pad = pad, bias_term = False, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	bn_conv = L.BatchNorm(conv, use_global_stats = 1)
	scale_conv = L.Scale(bn_conv, bias_term = True)
	return conv, bn_conv, scale_conv

def conv_bn_scale_relu(bottom, nout, ks, stride, pad):
	conv, bn_conv, scale_conv = res_uint(bottom, nout, ks, stride, pad)
	return conv, bn_conv, scale_conv, L.ReLU(scale_conv)

def res_unit_right(bottom, nout1, ks1, pad1, nout2, ks2, pad2, nout3, ks3, pad3)
	conv1, bn_conv1, scale_conv1, relu1 = conv_bn_scale_relu(bottom, nout1, ks1, stride1, pad1)
	conv2, bn_conv2, scale_conv2, relu2 = conv_bn_scale_relu(relu1, nout2, ks2, stride2, pad2)
	conv3, bn_conv3, scale_conv3 = conv_bn_scale(relu2, nout3, ks3, stride3, pad3)
	return conv1, bn_conv1, scale_conv1, relu1, conv2, bn_conv2, scale_conv2, relu2, conv3, bn_conv3, scale_conv3

def eltwise_relu(bottom1, bottom2, op=1):
	res_eltwise = L.Eltwise(bottom1, bottom2, P.Eltwise.SUM)
	return res_eltwise , L.ReLU(res_eltwise)

def resnet50-fcn(split):
	n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
            seed=1337)
    if split == 'train':
        pydata_params['sbdd_dir'] = '../data/sbdd/dataset'
        pylayer = 'SBDDSegDataLayer'
    else:
        pydata_params['voc_dir'] = '../data/pascal/VOC2011'
        pylayer = 'VOCSegDataLayer'

    n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1, n.bn_conv1, n.scale_conv1, n.conv1_relu = conv_bn_scale_relu(n.data, nout = 64, ks = 7, stride = 2, pad = 3)
    n.pool1 = max_pool(n.conv1_relu)

    n.res2a_branch2a, n.bn2a_branch2a, scale2a_branch2a, res2a_branch2a_relu = conv_bn_scale_relu(n.pool1, nout = 64, ks = 1, stride = 1, pad = 0)
	n.res2a_branch2b, n.bn2a_branch2b, scale2a_branch2b, res2a_branch2b_relu = conv_bn_scale_relu(n.pool1, nout = 64, ks = 3, stride = 1, pad = 1)
	n.res2a_branch2c, n.bn2a_branch2c, scale2a_branch2c = conv_bn_scale(n.res2a_branch2b_relu, nout = 256, ks = 1, stride=1, pad=0)
	n.res2a_branch1, n.bn2a_branch1, scale2a_branch1 = conv_bn_scale(n.pool1, nout = 256, ks = 1, stride=1, pad=0)
	n.res2a, n.res2a_relu = eltwise_relu(n.res2a_branch1, n.scale2a_branch1)

	n.res2b_branch2a, n.bn2b_branch2a, scale2b_branch2a, res2b_branch2a_relu = conv_bn_scale_relu(n.res2a_relu, nout = 64, ks = 1, stride = 1, pad = 0)
	n.res2b_branch2b, n.bn2b_branch2b, scale2b_branch2b, res2b_branch2b_relu = conv_bn_scale_relu(n.res2b_branch2a_relu, nout = 64, ks = 3, stride = 1, pad = 1)
	n.res2b_branch2c, n.bn2b_branch2c, scale2b_branch2c = conv_bn_scale(n.res2b_branch2b_relu, nout = 256, ks = 1, stride=1, pad=0)
	n.res2b, n.res2b_relu = eltwise_relu(n.res2a_relu, n.scale2b_branch2c)

	
	














