"""Run with: python -m unittest discover -s tests -p test_unipc_small_solve.py."""

import unittest
from unittest.mock import patch

import torch

from diffusers_helper.k_diffusion import uni_pc_fm as unipc


class SmallSolveTests(unittest.TestCase):
    def test_small_vandermonde_systems(self):
        for dtype in (torch.float32, torch.float64):
            for nodes in ([-0.8, 1.0], [-2.1, -0.8, 1.0]):
                x = torch.tensor(nodes, dtype=dtype)
                A = torch.stack([x ** k for k in range(len(nodes))])
                b = torch.arange(1, len(nodes) + 1, dtype=dtype) / 7
                actual = unipc._solve_small_linear_system(A, b)
                torch.testing.assert_close(actual, torch.linalg.solve(A, b))
                torch.testing.assert_close(A @ actual, b)

    def test_real_sampler_systems_and_trajectory(self):
        # Capture the actual matrices produced by shifted FramePack schedules,
        # and compare every step of a deterministic nonlinear toy-model run.
        noise = torch.linspace(-1, 1, 64).reshape(1, 4, 4, 4)
        model = lambda x, t: x.tanh() * 0.25 + t.reshape(-1, 1, 1, 1) * 0.1
        for variant in ('bh1', 'bh2'):
            for steps in (5, 25, 50):
                for shift in (1.0, 3.0, 7.0):
                    t = torch.linspace(1, 0, steps + 1)
                    sigmas = shift * t / (1 + (shift - 1) * t)
                    reference_steps, actual_steps = [], []

                    def checked_solve(A, b):
                        actual = unipc._solve_small_linear_system(A, b)
                        torch.testing.assert_close(actual, torch.linalg.solve(A, b), rtol=1e-4, atol=1e-6)
                        torch.testing.assert_close(A @ actual, b, rtol=1e-4, atol=1e-6)
                        return actual

                    def run(snapshots):
                        return unipc.sample_unipc(
                            model, noise.clone(), sigmas, extra_args={}, disable=True,
                            variant=variant,
                            callback=lambda d: snapshots.append(d['x'].clone()),
                        )

                    reference = run(reference_steps)
                    with patch.object(unipc, '_solve_linear_system', checked_solve):
                        actual = run(actual_steps)
                    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-6)
                    for actual_step, reference_step in zip(actual_steps, reference_steps):
                        torch.testing.assert_close(actual_step, reference_step, rtol=1e-4, atol=1e-6)

    def test_cpu_retains_native_solver(self):
        A, b = torch.eye(2), torch.ones(2)
        with patch.object(torch.linalg, 'solve', wraps=torch.linalg.solve) as native:
            unipc._solve_linear_system(A, b)
            native.assert_called_once_with(A, b)

    def test_npu_dispatch_without_native_solve(self):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            self.skipTest('TorchNPU is not installed')
        if not torch.npu.is_available():
            self.skipTest('No NPU available')
        x = torch.tensor([-2.1, -0.8, 1.0])
        A = torch.stack([x ** k for k in range(3)])
        b = torch.tensor([0.5, 0.25, 0.125])
        expected = torch.linalg.solve(A, b)
        with patch.object(torch.linalg, 'solve', side_effect=AssertionError('native solve called')):
            actual = unipc._solve_linear_system(A.to('npu:0'), b.to('npu:0'))
        self.assertEqual(actual.device.type, 'npu')
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
