import mxnet as mx

class Accuracy(mx.metric.EvalMetric):

    def __init__(self, index, name):
        super(Accuracy, self).__init__('accuracy[%s]' % name)
        if not isinstance(index, list):
            index = [index]
        self.index = index

    def update(self, labels, preds):
        # mx.metric.check_label_shapes(labels, preds)

        for index in self.index:
            label = labels[index].asnumpy().astype('int32')
            assert label.ndim in (1, 2)
            if label.ndim == 1:
                pred_label = mx.nd.argmax_channel(preds[index]).asnumpy().astype('int32')
            else:
                pred_label = (preds[index].asnumpy() > 0.5).astype('int32')

            # mx.metric.check_label_shapes(label, pred_label)

            if label.ndim == 1:
                mask = label >= 0
                label = label[mask]
                pred_label = pred_label[mask]

            # mx.metric.check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label == label).sum()
            self.num_inst += pred_label.size