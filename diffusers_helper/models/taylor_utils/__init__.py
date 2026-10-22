import torch
import math
from typing import Dict, Tuple


@torch.compile
def quantize_to_int8(tensor: torch.Tensor) -> tuple:

    tensor_flat = tensor.reshape(-1)
    
    min_val = torch.amin(tensor_flat)
    max_val = torch.amax(tensor_flat)
    
    is_min_eq_max = torch.eq(min_val, max_val)
    is_min_zero = torch.eq(min_val, 0)
    
    if torch.any(is_min_eq_max):
        if torch.any(is_min_zero):
            return torch.zeros_like(tensor, dtype=torch.int8), torch.tensor(1.0), torch.tensor(0)
        else:
            scale = torch.maximum(torch.neg(min_val), max_val) / 127.0
            zero_point = torch.tensor(0, dtype=torch.int32)
            fill_value = torch.where(min_val < 0, torch.tensor(-127), torch.tensor(127))
            return torch.full_like(tensor, fill_value, dtype=torch.int8), scale, zero_point
    
    scale = (max_val - min_val) / 255.0
    zero_point = torch.round(-min_val / scale) - 128
    zero_point = torch.clamp(zero_point, -128, 127).to(torch.int32)
    
    quantized_tensor = torch.round(tensor / scale + zero_point).to(torch.int8)
    
    return quantized_tensor, scale, zero_point

@torch.compile
def dequantize_from_int8(quantized_tensor: tuple) -> torch.Tensor:

    q_tensor, scale, zero_point = quantized_tensor
    return scale * (q_tensor.float() - zero_point)


@torch.compile
def quantize_to_int16(tensor: torch.Tensor) -> tuple:
    tensor_flat = tensor.reshape(-1)

    # Get the minimum and maximum values
    min_val = torch.amin(tensor_flat)
    max_val = torch.amax(tensor_flat)

    # Check if min == max, which means there's no variance in the tensor
    is_min_eq_max = torch.eq(min_val, max_val)
    is_min_zero = torch.eq(min_val, 0)

    # Handle edge case where min == max
    if torch.any(is_min_eq_max):
        if torch.any(is_min_zero):
            return torch.zeros_like(tensor, dtype=torch.int16), torch.tensor(1.0), torch.tensor(0)
        else:
            # Scale adjusted for 16-bit range
            scale = torch.maximum(torch.neg(min_val), max_val) / 32767.0  # Max value for int16
            zero_point = torch.tensor(0, dtype=torch.int32)
            fill_value = torch.where(min_val < 0, torch.tensor(-32767), torch.tensor(32767))  # Range for int16
            return torch.full_like(tensor, fill_value, dtype=torch.int16), scale, zero_point

    # Normal case: Calculate scale and zero_point
    scale = (max_val - min_val) / 65535.0  # 65535 for int16
    zero_point = torch.round(-min_val / scale) - 32768  # Zero-point adjusted for int16
    zero_point = torch.clamp(zero_point, -32768, 32767).to(torch.int32)  # Clamp within int16 range

    # Perform quantization
    quantized_tensor = torch.round(tensor / scale + zero_point).to(torch.int16)

    return quantized_tensor, scale, zero_point


@torch.compile
def dequantize_from_int16(quantized_tensor: tuple) -> torch.Tensor:
    q_tensor, scale, zero_point = quantized_tensor
    # Dequantization step, using int16 values
    return scale * (q_tensor.float() - zero_point)


# def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
#     """
#     Compute derivative approximation
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
#     #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

#     updated_taylor_factors = {}
#     updated_taylor_factors[0] = quantize_to_int8(feature)
    
#     for i in range(cache_dic['max_order']):
#         if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            
#             next_derivative = (dequantize_from_int8(updated_taylor_factors[i]) - dequantize_from_int8(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i])) / difference_distance
#             updated_taylor_factors[i + 1] = quantize_to_int8(next_derivative)
#         else:
#             break
    
#     cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

# def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
#     """
#     Compute Taylor expansion error
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     x = current['step'] - current['activated_steps'][-1]
#     output = 0

#     for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
#         output += (1 / math.factorial(i)) * dequantize_from_int8(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) * (x ** i)
    
#     return output

# def taylor_cache_init(cache_dic: Dict, current: Dict):
#     """
#     Initialize Taylor cache, expanding storage areas for Taylor series derivatives
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     if current['step'] == 0:
#         cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}




# def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
#     """
#     Compute derivative approximation
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
#     #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

#     updated_taylor_factors = {}
#     updated_taylor_factors[0] = quantize_to_int8(feature)
    
#     for i in range(cache_dic['max_order']):
#         if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            
#             next_derivative = (dequantize_from_int8(updated_taylor_factors[i]) - dequantize_from_int8(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i])) / difference_distance
#             updated_taylor_factors[i + 1] = quantize_to_int8(next_derivative)
#         else:
#             break
    
#     cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

# def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
#     """
#     Compute Taylor expansion error
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     x = current['step'] - current['activated_steps'][-1]
#     output = 0

#     for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
#         output += (1 / math.factorial(i)) * dequantize_from_int8(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) * (x ** i)
    
#     return output

# def taylor_cache_init(cache_dic: Dict, current: Dict):
#     """
#     Initialize Taylor cache, expanding storage areas for Taylor series derivatives
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     if current['step'] == 0:
#         cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}




# def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
#     """
#     Compute derivative approximation
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

#     updated_taylor_factors = {}
#     updated_taylor_factors[0] = feature

#     for i in range(cache_dic['max_order']):
#         if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
#             updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
#         else:
#             break
    
#     cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def derivative_approximation_terc(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """Update Taylor coefficients from full-compute residuals."""
    # feature is the residual across all Transformer blocks.
    previous = cache_dic['cache'][-1]
    updated = {0: feature}

    # The first observation supplies only the zeroth-order term.
    if previous and cache_dic['max_order'] > 0:
        steps = current['activated_steps']
        distance = steps[-1] - steps[-2]
        # Use the original finite-difference recurrence.
        for i in range(cache_dic['max_order']):
            if i not in previous:
                break
            updated[i + 1] = (updated[i] - previous[i]) / distance

    cache_dic['cache'][-1] = updated


# def taylor_formula(module_dict: Dict, distance: int) -> torch.Tensor: 
#     """
#     Compute Taylor expansion error.
#     :param cache_dic: Cache dictionary.
#     :param current: Current step information.
#     """

#     output = 0

#     for i in range(len(module_dict)):
#         output += (1 / math.factorial(i)) * module_dict[i] * (distance ** i)
    
#     return output


def taylor_formula_terc(module_dict: Dict, distance: int) -> torch.Tensor:
    """Predict a residual without updating the cache."""
    output = 0
    for i in range(len(module_dict)):
        output += (1 / math.factorial(i)) * module_dict[i] * (distance ** i)
    return output


# def taylor_cache_init(cache_dic: Dict, current: Dict):
#     """
#     Initialize Taylor cache, expanding storage areas for Taylor series derivatives
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     if current['step'] == 0:
#         cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}


def taylor_cache_init_terc(cache_dic: Dict, current: Dict):
    """Reset at step zero; retain later coefficients for differencing."""
    if current['step'] == 0:
        cache_dic['cache'][-1] = {}
