# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import copy
import torch
from schema import Schema, And, Or, Optional
from torch.serialization import validate_cuda_device

from SKDX.compression.utils.config_validation import CompressorSchema
from SKDX.compression.compressor import QuantType, QuantGrad, Quantizer

# for debug
import numpy as np
import sys

__all__ = ['NaiveQuantizer', 'QAT_Quantizer', 'DoReFaQuantizer', 'BNNQuantizer']

logger = logging.getLogger(__name__)

class NaiveQuantizer(Quantizer):
    """quantize weight to 8 bits
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        self.layer_scale = {}

    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            Optional('quant_types'): ['weight'],
            Optional('quant_bits'): Or(8, {'weight': 8}),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        new_scale = weight.abs().max() / 127
        scale = max(self.layer_scale.get(wrapper.name, 0), new_scale)
        self.layer_scale[wrapper.name] = scale
        orig_type = weight.type()  # TODO: user layer
        weight = weight.div(scale).type(torch.int8).type(orig_type).mul(scale)
        wrapper.module.weight = weight
        return weight

def update_ema(biased_ema, value, decay, step):
    """
    calculate biased stat and unbiased stat in each step using exponential moving average method

    Parameters
    ----------
    biased_ema : float
        previous stat value
    value : float
        current stat value
    decay : float
        the weight of previous stat value, larger means smoother curve
    step : int
        current step

    Returns
    -------
    float, float
    """
    biased_ema = biased_ema * decay + (1 - decay) * value
    unbiased_ema = biased_ema / (1 - decay ** step)  # Bias correction
    return biased_ema, unbiased_ema

def minmax_ema(biased_value, value, averaging_constant=0.01):
    """
    TensorRT Implementation

    Parameters
    ----------
    biased_value : float
        previous stat value
    value : float
        current stat value

    Returns
    -------
    float
    """
    #averaging_constant = 0.01
    #min_val_cur, max_val_cur = torch._aminmax(x)
    #min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
    #max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
    unbiased_ema = value + averaging_constant * (value - biased_value)
    return unbiased_ema

def update_quantization_param(bits, rmin, rmax, asymmetric):
    """
    calculate the `zero_point` and `scale` for parameters.

    Parameters
    ----------
    bits : int
        quantization bits length
    rmin : float
        min value of real value
    rmax : float
        max value of real value

    Returns
    -------
    float, float
    """

    if not asymmetric:
        # First determine the scale.
        threshold = torch.Tensor([(1 << (bits - 1)) - 1.0]).to(rmin.device)
        scale = rmax / threshold
        #print("scale", scale.shape)

        # Zero-point is fixed.
        initial_zero_point = 0

        # Now we need to nudge the zero point to be an integer
        if initial_zero_point == 0:
            nudged_zero_point = torch.zeros_like(scale)
    else:
        # extend the [min, max] interval to ensure that it contains 0.
        # Otherwise, we would not meet the requirement that 0 be an exactly
        # representable value.
        rmin = torch.min(rmin, torch.Tensor([0]).to(rmin.device))
        rmax = torch.max(rmax, torch.Tensor([0]).to(rmin.device))
        qmin = torch.Tensor([0]).to(rmin.device)
        qmax = torch.Tensor([(1 << bits) - 1]).to(rmin.device)

        # First determine the scale.
        scale = (rmax - rmin) / (qmax - qmin)

        # Zero-point computation.
        initial_zero_point = qmin - rmin / scale

        # Now we need to nudge the zero point to be an integer
        if initial_zero_point < qmin:
            nudged_zero_point = qmin
        elif initial_zero_point > qmax:
            nudged_zero_point = qmax
        else:
            nudged_zero_point = torch.round(initial_zero_point)

    return scale, nudged_zero_point


def get_bits_length(config, quant_type):
    if isinstance(config["quant_bits"], int):
        return config["quant_bits"]
    else:
        return config["quant_bits"].get(quant_type)


class QAT_Quantizer(Quantizer):
    """Quantizer defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """

    def __init__(self, model, config_list, optimizer=None):
        """
        Parameters
        ----------
        layer : LayerInfo
            the layer to quantize
        config_list : list of dict
            list of configurations for quantization
            supported keys for dict:
                - quant_types : list of string
                    type of quantization you want to apply, currently support 'weight', 'input', 'output'
                - quant_bits : int or dict of {str : int}
                    bits length of quantization, key is the quantization type, value is the length, eg. {'weight', 8},
                    when the type is int, all quantization types share same bits length
                - quant_start_step : int
                    disable quantization until model are run by certain number of steps, this allows the network to enter a more stable
                    state where activation quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0
                - op_types : list of string
                    types of nn.module you want to apply quantization, eg. 'Conv2d'
        """
        super().__init__(model, config_list, optimizer)
        self.steps = 1
        modules_to_compress = self.get_modules_to_compress()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print("modules_to_compress ", modules_to_compress)
        #print("config_list ", config_list)
        for layer, config in modules_to_compress:
            layer.module.register_buffer("zero_point", None)
            layer.module.register_buffer("scale", None)
            if "output" in config.get("quant_types", []):
                #layer.module.register_buffer('ema_decay', torch.Tensor([0.99]))
                #layer.module.register_buffer('tracked_min_biased', torch.zeros(1))
                layer.module.register_buffer('tracked_min', torch.zeros(1).to(device))
                #layer.module.register_buffer('tracked_max_biased', torch.zeros(1))
                layer.module.register_buffer('tracked_max', torch.zeros(1).to(device))

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list of dict
            List of configurations
        """
        schema = CompressorSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight', 'output']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('weight'): And(int, lambda n: 0 < n < 32),
                Optional('output'): And(int, lambda n: 0 < n < 32),
            })),
            Optional('quant_start_step'): And(int, lambda n: n >= 0),
            Optional('op_types'): [str],
            Optional('op_names'): [str],
            Optional('asymmetric'): And(int, lambda n: 0 <= n <= 1)
        }], model, logger)

        schema.validate(config_list)

    def _quantize(self, bits, op, real_val):
        """
        quantize real value.

        Parameters
        ----------
        bits : int
            quantization bits length
        op : torch.nn.Module
            target module
        real_val : float
            real value to be quantized

        Returns
        -------
        float
        """
        #print("zero_point", op.zero_point.shape)
        #print("real_val", real_val.shape)
        #print("scale", op.scale.shape)
        #print("asymmetric", asymmetric)

        op.zero_point = op.zero_point.to(real_val.device)
        op.scale = op.scale.to(real_val.device)
        transformed_val = op.zero_point + real_val / (op.scale + 1e-31)

        qmax = (float)(1 << (bits - 1)) - 1.0
        qmin = -qmax
            
        clamped_val = torch.clamp(transformed_val, qmin, qmax)
        quantized_val = torch.round(clamped_val)

        return quantized_val

    def _dequantize(self, op, quantized_val):
        """
        dequantize quantized value.
        Because we simulate quantization in training process, all the computations still happen as float point computations, which means we
        first quantize tensors then dequantize them. For more details, please refer to the paper.

        Parameters
        ----------
        op : torch.nn.Module
            target module
        quantized_val : float
            quantized_val value to be dequantized

        Returns
        -------
        float
        """

        real_val = op.scale * (quantized_val - op.zero_point)
        real_val = real_val
        return real_val

    def quantize_weight(self, wrapper, **kwargs):
        config = wrapper.config
        module = wrapper.module
        
        # new added code for quantize the rnn module
        """
        if hasattr(wrapper.module, 'old_weight_ih_l0') and wrapper.module.old_weight_ih_l0 is not None:
            weight = copy.deepcopy(wrapper.module.old_weight_ih_l0.data)
            #print("old_weight_ih_l0", weight.shape)
        elif hasattr(wrapper.module, 'old_weight_hh_l0') and wrapper.module.old_weight_hh_l0 is not None:
            weight = copy.deepcopy(wrapper.module.old_weight_hh_l0.data)
            #print("old_weight_hh_l0", weight.shape)
        else:
            weight = copy.deepcopy(wrapper.module.old_weight.data)
        """
        weight = copy.deepcopy(wrapper.module.old_weight.data)

        weight_bits = get_bits_length(config, 'weight')
        quant_start_step = config.get('quant_start_step', 0)
        assert weight_bits >= 1, "quant bits length should be at least 1"

        asymmetric = config["asymmetric"]
        assert (asymmetric == 0 or asymmetric == 1), "value of asymmetric should be 0 or 1"

        if quant_start_step > self.steps:
            return weight

        # if bias exists, quantize bias to uint32
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None and type(wrapper.module.bias) is not bool:
            bias = wrapper.module.bias.data
            bias_bits = 32
            
            if not asymmetric:
                rmax = torch.abs(bias)
                rmin = -rmax
            else:
                rmin = torch.min(bias)
                rmax = torch.max(bias)

            module.scale, module.zero_point = update_quantization_param(bias_bits, rmin, rmax, asymmetric)
            bias = self._quantize(bias_bits, module, bias)
            bias = self._dequantize(module, bias)
            wrapper.module.bias.data = bias
        """
        elif type(wrapper.module.bias) is bool and wrapper.module.bias == True:
            bias_ih_l0 = wrapper.module.bias_ih_l0.data
            bias_hh_l0 = wrapper.module.bias_hh_l0.data
            bias_bits = 32

            if not asymmetric:
                rmax_ih = torch.abs(bias_ih_l0)
                rmin_ih = -rmax_ih
                rmax_hh = torch.abs(bias_hh_l0)
                rmin_hh = -rmax_hh
            else:
                rmin_ih = torch.min(bias_ih_l0)
                rmax_ih = torch.max(bias_ih_l0)
                rmin_hh = torch.min(bias_hh_l0)
                rmax_hh = torch.max(bias_hh_l0)

            module.scale_ih, module.zero_point_ih = update_quantization_param(bias_bits, rmin_ih, rmax_ih, asymmetric)
            module.scale_hh, module.zero_point_hh = update_quantization_param(bias_bits, rmin_hh, rmax_hh, asymmetric)
            bias_ih_l0 = self._quantize(bias_bits, module, bias_ih_l0)
            bias_ih_l0 = self._dequantize(module, bias_ih_l0)
            bias_hh_l0 = self._quantize(bias_bits, module, bias_hh_l0)
            bias_hh_l0 = self._dequantize(module, bias_hh_l0)
            wrapper.module.bias_ih_l0.data = bias_ih_l0
            wrapper.module.bias_hh_l0.data = bias_hh_l0
        """

        # quantize weight
        # new added code for quantize the rnn module
        """
        if hasattr(wrapper.module, 'old_weight_ih_l0') and wrapper.module.old_weight_ih_l0 is not None:
            weight_shape   = weight.shape
            rmax       = torch.max(abs(weight), dim=1, keepdim=True)[0]
            rmin           = -rmax

        elif hasattr(wrapper.module, 'old_weight_hh_l0') and wrapper.module.old_weight_ih_l0 is not None:
            weight_shape   = weight.shape
            rmax       = torch.max(abs(weight), dim=1, keepdim=True)[0]
            rmin           = -rmax
        else:
        """
        weight_shape   = weight.shape
        if not asymmetric:
            if len(weight_shape) > 3:
                rmax       = torch.max(torch.max(torch.max(abs(weight), dim=3, keepdim=True)[0], dim=2, keepdim=True)[0], dim=1, keepdim=True)[0]
            else:
                rmax       = torch.max(abs(weight), dim=1, keepdim=True)[0]
            rmin           = -rmax
        else:
            NotImplemented
        
        #print("weight", weight.shape, "rmax weight", rmax.shape)
        module.scale, module.zero_point = update_quantization_param(weight_bits, rmin, rmax, asymmetric)
        weight = self._quantize(weight_bits, module, weight)
        weight = self._dequantize(module, weight)

        wrapper.module.weight = weight
        # new added code for quantize the rnn module
        """
        if hasattr(wrapper.module, 'old_weight_ih_l0') and wrapper.module.old_weight_ih_l0 is not None:
            wrapper.module.weight_ih_l0 = weight
        elif hasattr(wrapper.module, 'old_weight_hh_l0') and wrapper.module.old_weight_hh_l0 is not None:
            wrapper.module.weight_hh_l0 = weight
        else:
            wrapper.module.weight = weight
        """
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        config = wrapper.config
        module = wrapper.module
        output_bits = get_bits_length(config, 'output')
        quant_start_step = config.get('quant_start_step', 0)
        assert output_bits >= 1, "quant bits length should be at least 1"

        asymmetric = config["asymmetric"]
        assert (asymmetric == 0 or asymmetric == 1), "value of asymmetric should be 0 or 1"

        if quant_start_step > self.steps:
            return output

        if not asymmetric:
            current_max = torch.max(output)
            current_min = -current_max
        else:
            current_max = torch.max(output)
            current_min = torch.min(output)
        """
        module.tracked_min_biased, module.tracked_min = update_ema(module.tracked_min_biased, current_min,
                                                                   module.ema_decay, self.steps)
        module.tracked_max_biased, module.tracked_max = update_ema(module.tracked_max_biased, current_max,
                                                                   module.ema_decay, self.steps)
        """
        if module.tracked_min == 0:
            module.tracked_min = current_min
        elif module.tracked_max == 0:
            module.tracked_max = current_max
        else:
            module.tracked_min = minmax_ema(module.tracked_min, current_min)
            module.tracked_max = minmax_ema(module.tracked_max, current_max)

        module.scale, module.zero_point = update_quantization_param(output_bits, module.tracked_min, module.tracked_max, asymmetric)
        out = self._quantize(output_bits, module, output)
        out = self._dequantize(module, out)
        return out

    def fold_bn(self, config, **kwargs):
        # TODO simulate folded weight
        pass

    def step_with_optimizer(self):
        """
        override `compressor` `step` method, quantization only happens after certain number of steps
        """
        self.steps += 1


class DoReFaQuantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list of dict
            List of configurations
        """
        schema = CompressorSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('weight'): And(int, lambda n: 0 < n < 32)
            })),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight_bits = get_bits_length(wrapper.config, 'weight')
        weight = weight.tanh()
        weight = weight / (2 * weight.abs().max()) + 0.5
        weight = self.quantize(weight, weight_bits)
        weight = 2 * weight - 1
        wrapper.module.weight = weight
        # wrapper.module.weight.data = weight
        return weight

    def quantize(self, input_ri, q_bits):
        scale = pow(2, q_bits) - 1
        output = torch.round(input_ri * scale) / scale
        return output


class ClipGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type):
        if quant_type == QuantType.QUANT_OUTPUT:
            grad_output[torch.abs(tensor) > 1] = 0
        return grad_output


class BNNQuantizer(Quantizer):
    """Binarized Neural Networks, as defined in:
    Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1
    (https://arxiv.org/abs/1602.02830)
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        self.quant_grad = ClipGrad

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list of dict
            List of configurations
        """
        schema = CompressorSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight', 'output']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('weight'): And(int, lambda n: 0 < n < 32),
                Optional('output'): And(int, lambda n: 0 < n < 32),
            })),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight = torch.sign(weight)
        # remove zeros
        weight[weight == 0] = 1
        wrapper.module.weight = weight
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        out = torch.sign(output)
        # remove zeros
        out[out == 0] = 1
        return out
