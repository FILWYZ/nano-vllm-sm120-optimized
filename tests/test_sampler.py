import unittest

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class TestSamplingParams(unittest.TestCase):
    def test_zero_temperature_enables_greedy(self):
        self.assertEqual(SamplingParams(temperature=0).temperature, 0)

    def test_negative_temperature_is_rejected(self):
        with self.assertRaises(AssertionError):
            SamplingParams(temperature=-0.1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler().cuda()

    def test_greedy_matches_argmax(self):
        logits = torch.tensor(
            [[1.0, 4.0, 2.0], [7.0, 3.0, 5.0]], device="cuda"
        )
        temperatures = torch.zeros(2, device="cuda")

        actual = self.sampler(logits, temperatures, True, True)

        torch.testing.assert_close(actual, logits.argmax(dim=-1))
        self.assertEqual(len(self.sampler.noise_buffers), 0)

    def test_mixed_batch_keeps_greedy_row_exact(self):
        torch.manual_seed(0)
        logits = torch.tensor(
            [[1.0, 9.0, 2.0], [1.0, 2.0, 3.0]], device="cuda"
        )
        temperatures = torch.tensor([0.0, 0.8], device="cuda")

        actual = self.sampler(logits, temperatures, False, True)

        self.assertEqual(actual[0].item(), 1)

    def test_stochastic_noise_buffer_is_reused(self):
        logits = torch.randn(4, 32, device="cuda", dtype=torch.float16)
        temperatures = torch.full((4,), 0.6, device="cuda")
        self.sampler(logits, temperatures, False, False)
        buffer = next(iter(self.sampler.noise_buffers.values()))
        pointer = buffer.data_ptr()

        self.sampler(logits, temperatures, False, False)

        self.assertEqual(len(self.sampler.noise_buffers), 1)
        self.assertEqual(
            next(iter(self.sampler.noise_buffers.values())).data_ptr(), pointer
        )


if __name__ == "__main__":
    unittest.main()
