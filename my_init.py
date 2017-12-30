import mxnet as mx
import numpy as np
import os
import re


class Init(mx.init.Xavier):

    def __init__(self, *args, **kargs):
        super(Init, self).__init__(*args, **kargs)

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')

        if name.endswith('lambda'):
            self._init_zero(name, arr)
        elif name.endswith('_zero'):
            self._init_zero(name, arr)
        elif name.endswith('_one'):
            self._init_one(name, arr)
        elif name.endswith('gradscale_scale'):
            self._init_gradscale(name, arr)
        elif name.startswith('sum'):
            self._init_gamma(name, arr)
        elif 'im2col' in name and name.endswith('weight'):
            self._init_im2col(name, arr)
        elif 'rnn' in name and name.endswith('_init_c'):
            self._init_zero(name, arr)
        elif 'rnn' in name and name.endswith('_init_h'):
            self._init_zero(name, arr)
        elif name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif 'rnn' in name and name.endswith('gamma'):
            self._init_rnn_gamma(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)

    def _init_one(self, _, arr):
        arr[:] = 1

    def _init_gradscale(self, name, arr):
        arr[:] = {
            'gesture_gradscale_scale': 0,
            'bottleneck_gradscale_scale': 0,
            'subject_gradscale_scale': 1,
            'subject_confusion_gradscale_scale': 0,
        }[name]

    def _init_rnn_gamma(self, _, arr):
        arr[:] = 0.1

    def _init_h2h(self, _, arr):
        assert len(arr.shape) == 2
        assert arr.shape[0] == arr.shape[1] * 4
        n = arr.shape[1]
        eye = np.eye(n)
        for i in range(4):
            arr[i * n:(i + 1) * n] = eye

    def _init_im2col(self, _, arr):
        assert len(arr.shape) == 4
        assert arr.shape[0] == arr.shape[1] * arr.shape[2] * arr.shape[3]
        arr[:] = np.eye(arr.shape[0]).reshape(arr.shape)

'''
class Load(mx.init.Load):

    def __init__(self, params, *args, **kargs):
        self.ignore = kargs.pop('ignore', [])
        mod = kargs.pop('mod')
        if not os.path.exists(params) and os.path.exists(os.path.join(ROOT, params)):
            params = os.path.join(ROOT, params)
        super(Load, self).__init__(params, *args, **kargs)
        for name in list(self.param):
            for ignore in self.ignore:
                if re.match(ignore, name):
                    logger.info('Ignore param {}', name)
                    del self.param[name]
                    break
            if mod.adabn and (name.endswith('moving_mean')
                              or name.endswith('moving_var')):
                del self.param[name]
            if mod.rnn and 'rnn' in name and (name.endswith('_init_c')
                                                or name.endswith('_init_h')):
                del self.param[name]

    def __call__(self, name, arr):
        if name in self.param and ('gamma' in name or 'beta' in name):
            self.param[name] = self.param[name].reshape(arr.shape)
        return super(Load, self).__call__(name, arr)
'''