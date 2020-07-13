import torch
import unittest
from gpu_everything import preprocessing


class TestAugmentations(unittest.TestCase):
    def setUp(self) -> None:
        self.random_tensor = torch.rand(size=(10, 10))
        super().__init__()

    def test_z_score_norm_statistics(self):
        z_score_norm = preprocessing.ZScoreNorm()
        normed_tensor = z_score_norm(self.random_tensor)
        self.assertAlmostEqual(float(torch.mean(normed_tensor)), 0.0, places=5)
        self.assertAlmostEqual(float(torch.std(normed_tensor)), 1.0, places=5)

    def test_z_score_norm_identity(self):
        z_score_norm = preprocessing.ZScoreNorm()
        std = torch.std(self.random_tensor)
        mean = torch.mean(self.random_tensor)
        normed_tensor = z_score_norm(self.random_tensor)
        self.assertTrue(torch.allclose(normed_tensor * std + mean, self.random_tensor))


if __name__ == "__main__":
    unittest.main()

