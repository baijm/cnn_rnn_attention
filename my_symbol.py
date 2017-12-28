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
              batch_norm=False):
    """ get VGG16 symbol
    Parameters
    ----------------------------
    data: mx.symbol
        input
    num_classes: int
        number of classification classes
    batch_norm: bool (default = False)
        whether to use batch normalization

    Return
    -----------------------------
    {layer_name -> symbol, 'loss' -> []}
    """

    layers, filters = ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
    net = data
    out = {}

    # convolutional blocks
    for i, num in enumerate(layers):
        for j in range(num):
            net = mx.sym.Convolution(data=net,
                                     kernel=(3, 3), pad=(1, 1),
                                     num_filter=filters[i],
                                     name="conv%s_%s" % (i + 1, j + 1)
                                     )
            out["conv%s_%s" % (i + 1, j + 1)] = net

            if batch_norm:
                net = mx.symbol.BatchNorm(data=net,
                                          name="bn%s_%s" % (i + 1, j + 1)
                                          )
                out["bn%s_%s" % (i + 1, j + 1)] = net

            net = mx.sym.Activation(data=net,
                                    act_type="relu",
                                    name="relu%s_%s" % (i + 1, j + 1)
                                    )
            out["relu%s_%s" % (i + 1, j + 1)] = net

        net = mx.sym.Pooling(data=net,
                             pool_type="max",
                             kernel=(2, 2),
                             stride=(2, 2),
                             name="pool%s" % (i + 1))
        out["pool%s" % (i + 1)] = net

    net = mx.sym.Flatten(data=net, name="flatten")
    out["flatten"] = net

    # fully connected layers
    net = mx.sym.FullyConnected(data=net, num_hidden=4096, name="fc6")
    out["fc6"] = net
    net = mx.sym.Activation(data=net, act_type="relu", name="relu6")
    out["relu6"] = net
    net = mx.sym.Dropout(data=net, p=0.5, name="drop6")
    out["drop6"] = net

    net = mx.sym.FullyConnected(data=net, num_hidden=4096, name="fc7")
    out["fc7"] = net
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


# TODO : still not sure how to make the two losses in paper ... just return softmax now
def get_cnn(
        num_cls,
        for_training=False,
        **kargs):
    """ get model with only one CNN module (VGG16)
    Parameters
    ----------------------------
    num_cls : int
        number of classes
    for_training : bool
    kargs : {}
        optional parameters

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
        num_classes=num_cls
    )["softmax"]

    # TODO : still not sure how to make the two losses in paper ...
    feature = get_vgg16(
        data=net,
        num_classes=num_cls
    )["relu7"]
    loss = []


def get_cnn_rnn_attention(
        num_cls,
        for_training,
        rnn_dropout,
        rnn_hidden,
        rnn_window, # time steps ?
        **kargs):
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
    kargs: {}

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
    net = mx.symbol.Variable(name="data")
    # TODO : reshape, swapaxis, reshape (see symbol_overlap_feature_single_att1_loss_bjm.py : line 232-234)
    # TODO : batchnorm (see symbol_overlap_feature_single_att1_loss_bjm.py : line 238-243)

    """
    CNN module
    """
    feature = get_vgg16(
        data=net,
        num_classes=num_cls,
    )["relu7"]

    loss = []

    """
    RNN module
    """
    # TODO : reshape and swapaxis, split data into time steps (see symbol_overlap_feature_single_att1_loss_bjm.py : line 261-269)

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
    )

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
    )
    # alphas
    a = mx.symbol.SoftmaxActivation(
        name='atten_softmax_%d' % i,
        data=M
    )
    a = mx.sym.Reshape(
        data=a,
        shape=(-1, rnn_window, 1)
    )
    # r
    r = mx.symbol.broadcast_mul(
        name='atten_r_%d' % i,
        lhs=a,
        rhs=concat_feature
    )
    z = mx.sym.sum(data=r, axis=1)

    # loss_target is used only in training
    if for_training:
        feature = mx.symbol.Concat(
            *feature,
            dim=0
        )

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
    )
    att_gesture_branch_kargs['label'] = att_gesture_label
    att_gesture_branch_kargs['grad_scale'] = 0.1 / rnn_window

    att_gesture_softmax, att_gesture_fc = get_branch(
        for_training=for_training,
        name='gesture',  # m
        data=z,
        num_class=num_cls,
        return_fc=True,
        use_ignore=True,  # ???
        **att_gesture_branch_kargs
    )
    loss.insert(0, att_gesture_softmax)

    # TODO : self.net in symbol_overlap_feature_single_att1_loss_bjm.py : line 411
    net = loss[0] if len(loss) == 1 else mx.sym.Group(loss)

    # TODO : code below are from symbol_overlap_feature_single_att1_loss_bjm.py : line 426+
    net_inf = mx.symbol.FullyConnected(
        name='gesture_last_fc',
        data=z,
        num_hidden=num_cls,
        no_bias=False
    )
    net_inf = mx.symbol.SoftmaxActivation(net_inf)

    """
    things to return
    """
    return (net, net_inf)


