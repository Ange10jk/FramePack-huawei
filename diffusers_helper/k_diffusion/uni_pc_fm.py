# Better Flow Matching UniPC by Lvmin Zhang
# (c) 2025
# CC BY-SA 4.0
# Attribution-ShareAlike 4.0 International Licence


import math

import torch

from tqdm.auto import trange


def _solve_small_linear_system(matrix, rhs):
    """Solve UniPC's fixed 2x2/3x3 systems using NPU-supported tensor ops."""
    size = matrix.shape[0]

    if size == 2:
        a00, a01 = matrix[0].unbind()
        a10, a11 = matrix[1].unbind()
        b0, b1 = rhs.unbind()
        determinant = a00 * a11 - a01 * a10
        return torch.stack((
            (b0 * a11 - a01 * b1) / determinant,
            (a00 * b1 - b0 * a10) / determinant,
        ))

    if size == 3:
        a00, a01, a02 = matrix[0].unbind()
        a10, a11, a12 = matrix[1].unbind()
        a20, a21, a22 = matrix[2].unbind()
        b0, b1, b2 = rhs.unbind()

        determinant = (
            a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20)
        )
        determinant_0 = (
            b0 * (a11 * a22 - a12 * a21)
            - a01 * (b1 * a22 - a12 * b2)
            + a02 * (b1 * a21 - a11 * b2)
        )
        determinant_1 = (
            a00 * (b1 * a22 - a12 * b2)
            - b0 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * b2 - b1 * a20)
        )
        determinant_2 = (
            a00 * (a11 * b2 - b1 * a21)
            - a01 * (a10 * b2 - b1 * a20)
            + b0 * (a10 * a21 - a11 * a20)
        )
        return torch.stack((determinant_0, determinant_1, determinant_2)) / determinant

    raise ValueError(f'UniPC expected a 2x2 or 3x3 system, got {tuple(matrix.shape)}')


def _solve_linear_system(matrix, rhs):
    if (
        matrix.device.type == 'npu'
        and matrix.ndim == 2
        and rhs.ndim == 1
        and matrix.shape[0] == matrix.shape[1]
        and matrix.shape[0] in (2, 3)
    ):
        return _solve_small_linear_system(matrix, rhs)
    return torch.linalg.solve(matrix, rhs)


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


def _unipc_coefficients(hh, order, variant):
    """Compute i! * phi_(i+1)(hh) * hh / B(hh) without cancellation.

    For |hh| <= 1, evaluate phi with an 18th-degree Taylor polynomial.
    Away from zero the original recurrence is well conditioned for orders 1-3.
    Both branches use safe inputs, so torch.where needs no host synchronization.
    """
    small = hh.abs() <= 1
    series_h = torch.where(small, hh, torch.zeros_like(hh))
    recurrence_h = torch.where(small, -torch.ones_like(hh), hh)
    recurrence_expm1 = torch.expm1(recurrence_h)
    phi = recurrence_expm1 / recurrence_h - 1
    coefficients = []
    for i in range(1, order + 1):
        factorial = math.factorial(i)
        series = torch.full_like(hh, factorial / math.factorial(19 + i))
        for k in range(17, -1, -1):
            series = series * series_h + factorial / math.factorial(k + i + 1)
        recurrent = phi * factorial / recurrence_h
        coefficients.append(torch.where(small, series, recurrent))
        phi = phi / recurrence_h - 1 / math.factorial(i + 1)

    b = torch.stack(coefficients)
    if variant == 'bh2':
        # phi_1(h) = expm1(h)/h, with a finite limit of 1 at zero.
        phi1 = torch.full_like(hh, 1 / math.factorial(19))
        for k in range(17, -1, -1):
            phi1 = phi1 * series_h + 1 / math.factorial(k + 1)
        ratio = torch.where(small, phi1, recurrence_expm1 / recurrence_h)
        b = b / ratio
    elif variant != 'bh1':
        raise NotImplementedError('Bad variant!')
    return b


class FlowMatchUniPC:
    def __init__(self, model, extra_args, variant='bh1'):
        self.model = model
        self.variant = variant
        self.extra_args = extra_args

    def model_fn(self, x, t):
        return self.model(x, t, **self.extra_args)

    def update_fn(self, x, model_prev_list, t_prev_list, t, order):
        assert order <= len(model_prev_list)
        dims = x.dim()

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = - torch.log(t_prev_0)
        lambda_t = - torch.log(t)
        model_prev_0 = model_prev_list[-1]

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = - torch.log(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(t.new_ones(()))
        rks = torch.stack(rks)

        R = []

        hh = -h[0]
        h_phi_1 = torch.expm1(hh)

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError('Bad variant!')

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))

        R = torch.stack(R)
        b = _unipc_coefficients(hh, order, self.variant)

        use_predictor = len(D1s) > 0

        if use_predictor:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = b.new_tensor([0.5])
            else:
                rhos_p = _solve_linear_system(R[:-1, :-1], b[:-1])
        else:
            D1s = None
            rhos_p = None

        if order == 1:
            rhos_c = b.new_tensor([0.5])
        else:
            rhos_c = _solve_linear_system(R, b)

        x_t_ = expand_dims(t / t_prev_0, dims) * x - expand_dims(h_phi_1, dims) * model_prev_0

        if use_predictor:
            pred_res = torch.tensordot(D1s, rhos_p, dims=([1], [0]))
        else:
            pred_res = 0

        x_t = x_t_ - expand_dims(B_h, dims) * pred_res
        model_t = self.model_fn(x_t, t)

        if D1s is not None:
            corr_res = torch.tensordot(D1s, rhos_c[:-1], dims=([1], [0]))
        else:
            corr_res = 0

        D1_t = (model_t - model_prev_0)
        x_t = x_t_ - expand_dims(B_h, dims) * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    def sample(self, x, sigmas, callback=None, disable_pbar=False):
        order = min(3, len(sigmas) - 2)
        model_prev_list, t_prev_list = [], []
        for i in trange(len(sigmas) - 1, disable=disable_pbar):
            vec_t = sigmas[i].expand(x.shape[0])

            if i == 0:
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
            elif i < order:
                init_order = i
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, init_order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            else:
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)

            model_prev_list = model_prev_list[-order:]
            t_prev_list = t_prev_list[-order:]

            if callback is not None:
                callback({'x': x, 'i': i, 'denoised': model_prev_list[-1]})

        return model_prev_list[-1]


def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=False, variant='bh1'):
    assert variant in ['bh1', 'bh2']
    return FlowMatchUniPC(model, extra_args=extra_args, variant=variant).sample(noise, sigmas=sigmas, callback=callback, disable_pbar=disable)
