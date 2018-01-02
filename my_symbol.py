"""
modified from https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/vgg.py
simplified for VGG-16
"""
import mxnet as mx
import my_constant
from lstm import lstm_unroll


def get_branch(for_training,
               name,
               data,
               num_class,
               return_fc=False,
               fc_attr={},
               **kargs):
    """ append fc and softmax to layer, for training or testing purposes
    Parameters
    --------------------------
    for_training: bool
    name: str
        prefix of layer names
    data: mx.symbol
        input
    num_class: int
    return_fc: bool
        if fc should be returned along with softmax
    fc_attr: {}
        attributes of fc layer
    kargs: {}
        optional parameters

    Return
    ---------------------------
    (net, fc) if return_fc else net
    """

    net = data
    net = mx.symbol.FullyConnected(
        name=name + '_last_fc',
        data=net,
        num_hidden=num_class,
        no_bias=False,
        attr=dict(**fc_attr)
    )
    fc = net
    if for_training:
        net = mx.symbol.SoftmaxOutput(
            name=name + '_softmax',
            data=net,
            **kargs
        )
    else:
        net = mx.symbol.SoftmaxActivation(
            name=name + '_softmax',
            data=net
        )
    return (net, fc) if return_fc else net


def get_vgg16(data,
              num_classes,
              batch_norm=False,
              fix_till_relu7=False):
    """ get VGG16 symbol
    Parameters
    ----------------------------
    data: mx.symbol
        input
    num_classes: int
        number of classification classes
    batch_norm: bool (default = False)
        whether to use batch normalization
    fix_till_relu7 : bool (default = False)
        whether to set

    Return
    -----------------------------
    {layer_name -> symbol, 'loss' -> []}
    """

    # used to fix feature layers when training cnn+rnn+attention
    lr_attr = {'lr_mult': '0'}

    layers, filters = ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
    net = data
    out = {}

    # convolutional blocks
    for i, num in enumerate(layers):
        for j in range(num):
            if fix_till_relu7:
                net = mx.sym.Convolution(data=net,
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=filters[i],
                                         name="conv%s_%s" % (i + 1, j + 1),
                                         attr=lr_attr)
            else:
                net = mx.sym.Convolution(data=net,
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=filters[i],
                                         name="conv%s_%s" % (i + 1, j + 1))
            out["conv%s_%s" % (i + 1, j + 1)] = net

            if batch_norm:
                if fix_till_relu7:
                    net = mx.symbol.BatchNorm(data=net,
                                              name="bn%s_%s" % (i + 1, j + 1),
                                              attr=lr_attr)
                else:
                    net = mx.symbol.BatchNorm(data=net,
                                              name="bn%s_%s" % (i + 1, j + 1))

                out["bn%s_%s" % (i + 1, j + 1)] = net

            if fix_till_relu7:
                net = mx.sym.Activation(data=net,
                                        act_type="relu",
                                        name="relu%s_%s" % (i + 1, j + 1),
                                        attr=lr_attr)
            else:
                net = mx.sym.Activation(data=net,
                                        act_type="relu",
                                        name="relu%s_%s" % (i + 1, j + 1))
            out["relu%s_%s" % (i + 1, j + 1)] = net

        if fix_till_relu7:
            net = mx.sym.Pooling(data=net,
                                 pool_type="max",
                                 kernel=(2, 2),
                                 stride=(2, 2),
                                 name="pool%s" % (i + 1),
                                 attr=lr_attr)
        else:
            net = mx.sym.Pooling(data=net,
                                 pool_type="max",
                                 kernel=(2, 2),
                                 stride=(2, 2),
                                 name="pool%s" % (i + 1))
        out["pool%s" % (i + 1)] = net

    if fix_till_relu7:
        net = mx.sym.Flatten(data=net, name="flatten", attr=lr_attr)
    else:
        net = mx.sym.Flatten(data=net, name="flatten")

    out["flatten"] = net

    # fully connected layers
    if fix_till_relu7:
        net = mx.sym.FullyConnected(data=net, num_hidden=4096, name="fc6", attr=lr_attr)
    else:
        net = mx.sym.FullyConnected(data=net, num_hidden=4096, name="fc6")
    out["fc6"] = net

    if fix_till_relu7:
        net = mx.sym.Activation(data=net, act_type="relu", name="relu6", attr=lr_attr)
    else:
        net = mx.sym.Activation(data=net, act_type="relu", name="relu6")
    out["relu6"] = net

    if fix_till_relu7:
        net = mx.sym.Dropout(data=net, p=0.5, name="drop6", attr=lr_attr)
    else:
        net = mx.sym.Dropout(data=net, p=0.5, name="drop6")
    out["drop6"] = net

    if fix_till_relu7:
        net = mx.sym.FullyConnected(data=net, num_hidden=4096, name="fc7", attr=lr_attr)
    else:
        net = mx.sym.FullyConnected(data=net, num_hidden=4096, name="fc7")
    out["fc7"] = net

    if fix_till_relu7:
        net = mx.sym.Activation(data=net, act_type="relu", name="relu7", attr=lr_attr)
    else:
        net = mx.sym.Activation(data=net, act_type="relu", name="relu7")
    out["relu7"] = net

    net = mx.sym.Dropout(data=net, p=0.5, name="drop7")
    out["drop7"] = net

    # layers not initialized from ImageNet model
    net = mx.sym.FullyConnected(data=net, num_hidden=num_classes, name="fc8_new")
    out["fc8_new"] = net
    net = mx.sym.SoftmaxOutput(data=net, name="softmax")
    out["softmax"] = net

    # TODO : what this does ?
    # as in symbol_overlap_feature_single_att1_loss.py : get_feature
    out["loss"] = []

    return out


def get_cnn(
        num_cls,
        fix_till_relu7=False):
    """ get model with only one CNN module (VGG16)
    Parameters
    ----------------------------
    num_cls : int
        number of classes
    fix_till_relu7 : bool
        whether to fix layers till relu7

    Return
    ----------------------------
    mx.symbol for net
    """

    # input
    net = mx.symbol.Variable(name="data")
    # TODO : batchnorm (see symbol_overlap_feature_single_att1_loss_bjm.py : line 238-243)

    """
    CNN module
    """
    return get_vgg16(
        data=net,
        num_classes=num_cls,
        fix_till_relu7=fix_till_relu7
    )["softmax"]


def get_cnn_rnn_attention(
        num_cls,
        for_training,
        rnn_dropout,
        rnn_hidden,
        rnn_window,
        fix_till_relu7=False
):
    """ get model with CNN + RNN + attention
    Parameters
    ----------------------------
    num_cls: int
        number of classes
    for_training: bool
    rnn_dropout: float
        RNN dropout probability
    rnn_hidden: int
        number of hidden units of each RNN unit
    rnn_window: int
        number of timesteps
    fix_till_relu7: bool
        whether to fix CNN feature extracting part

    Return
    -----------------------------
    (mx.symbol for net, mx.symbol for loss)
    """
    """
    require from DataIter:
        data
        gesture_softmax_label
        att_gesture_softmax_label
    """

    # input
    net = mx.symbol.Variable(name="data") # (batch_size, rnn_windows * c, h, w)
    net = mx.symbol.Reshape(net, shape=(0, -1, # (batch_size, rnn_windows, c, h, w)
                                        my_constant.INPUT_CHANNEL,
                                        my_constant.INPUT_SIDE, my_constant.INPUT_SIDE))
    net = mx.symbol.SwapAxis(net, # (rnn_windows, batch_size, c, h, w)
                             dim1=0, dim2=1)
    net = mx.symbol.Reshape(net, shape=(-1, # (rnn_windows * batch_size, c, h, w)
                                        my_constant.INPUT_CHANNEL,
                                        my_constant.INPUT_SIDE, my_constant.INPUT_SIDE))
    # TODO : batchnorm (see symbol_overlap_feature_single_att1_loss_bjm.py : line 238-243)

    """
    CNN module
    """
    feature = get_vgg16(
        data=net,
        num_classes=num_cls,
        fix_till_relu7=fix_till_relu7
    )["relu7"]

    loss = []

    """
    RNN module
    """
    # split into time steps
    feature = mx.symbol.Reshape(
        feature,
        shape=(rnn_window, -1, my_constant.FEATURE_DIM)
    ) # (32, 1, 4096)

    feature = mx.symbol.SwapAxis(
        feature,
        dim1=0,
        dim2=1
    ) # (1, 32, 4096)

    # f_h(X) (output of LSTM)
    feature = lstm_unroll(
        prefix='',
        data=feature,
        num_rnn_layer=1,
        seq_len=rnn_window,
        num_hidden=rnn_hidden,
        dropout=rnn_dropout,
        bn=True
    )

    concat_feature = mx.sym.Reshape(
        data=mx.sym.Concat(
            *feature,
            dim=1
        ),
        shape=(-1, rnn_window, rnn_hidden)
    ) # (1, 32, 512)

    """
    attention module
    """
    M = []
    # weight_v = w^T
    weight_v = mx.sym.Variable('atten_v_bias', shape=(rnn_hidden, 1))
    # weight_u = W_h
    weight_u = mx.sym.Variable('atten_u_weight', shape=(rnn_hidden, rnn_hidden))
    for i in range(rnn_window):
        # feature1[i] = h_t
        tmp = mx.sym.dot(
            feature[i],
            weight_u,
            name='atten_u_%d' % i
        )
        # M_t
        tmp = mx.sym.Activation(
            tmp,
            act_type='tanh'
        )
        tmp = mx.sym.dot(
            tmp,
            weight_v,
            name='atten_v_%d' % i
        )
        M.append(tmp)

    M = mx.sym.Concat(
        *M,
        dim=1
    ) # (1, 32)

    # alphas
    a = mx.symbol.SoftmaxActivation(
        name='atten_softmax_%d' % i,
        data=M
    )
    a = mx.sym.Reshape(
        data=a,
        shape=(-1, rnn_window, 1)
    ) # (1, 32, 1)

    # r
    r = mx.symbol.broadcast_mul(
        name='atten_r_%d' % i,
        lhs=a,
        rhs=concat_feature
    ) # (1, 32, 512)

    z = mx.sym.sum(data=r, axis=1) # (1, 512)

    # loss_target is used only in training
    if for_training:
        feature = mx.symbol.Concat(
            *feature,
            dim=0
        ) # (32, 512)

        # loss_target
        gesture_branch_kargs = {}
        gesture_label = mx.symbol.Variable(
            name='att_gesture_softmax_label'
        )  # m
        gesture_label = mx.symbol.Reshape(
            mx.symbol.Concat(
                *[mx.symbol.Reshape(
                    gesture_label,
                    shape=(0, 1)
                )
                  for i in range(rnn_window)],
                dim=0
            ),
            shape=(-1,)
        )

        gesture_branch_kargs['label'] = gesture_label
        gesture_branch_kargs['grad_scale'] = 1 / rnn_window
        gesture_softmax, gesture_fc = get_branch(
            for_training=for_training,
            name='att_gesture',  # m
            data=feature,
            num_class=num_cls,
            return_fc=True,
            use_ignore=True,  # ???
            **gesture_branch_kargs
        )

        loss.append(gesture_softmax)


    # loss_attention is used in both training and testing
    att_gesture_branch_kargs = {}
    att_gesture_label = mx.symbol.Variable(
        name='gesture_softmax_label'
    )  # m

    att_gesture_label = mx.symbol.Reshape(
        mx.symbol.Concat(
            *[mx.symbol.Reshape(
                att_gesture_label,
                shape=(0, 1)
            )
              for i in range(1)
              ],
            dim=0
        ),
        shape=(-1,)
    ) # (1,)

    att_gesture_branch_kargs['label'] = att_gesture_label
    att_gesture_branch_kargs['grad_scale'] = 0.1 / rnn_window

    att_gesture_softmax, att_gesture_fc = get_branch( # (1, 200)
        for_training=for_training,
        name='gesture',  # m
        data=z, # (1, 512)
        num_class=num_cls,
        return_fc=True,
        use_ignore=True,  # ???
        **att_gesture_branch_kargs
    )

    loss.insert(0, att_gesture_softmax)


    net = loss[0] if len(loss) == 1 else mx.sym.Group(loss)
    return net

    # TODO : code below are from symbol_overlap_feature_single_att1_loss_bjm.py : line 426+
    # TODO : used in optimazation but keep it here
    #net_inf = mx.symbol.FullyConnected(
    #    name='gesture_last_fc',
    #    data=z,
    #    num_hidden=num_cls,
    #    no_bias=False
    #)
    #net_inf = mx.symbol.SoftmaxActivation(net_inf)



