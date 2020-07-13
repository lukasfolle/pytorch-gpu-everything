import torch
import unittest
from gpu_everything import augmentation


class TestAugmentations(unittest.TestCase):
    def setUp(self) -> None:
        self.random_tensor = torch.rand(size=(10, 10))
        super().__init__()

    def test_random_rotation_equal(self):
        random_rotation_aug = augmentation.RandomRotate90(dims=(0, 1), prob=1.0, max_number_rot=1)
        output_tensor = random_rotation_aug(self.random_tensor)
        expected_output_tensor = torch.rot90(self.random_tensor, 1, (0, 1))
        self.assertTrue(torch.equal(expected_output_tensor, output_tensor))

    def test_random_rotation_changed(self):
        random_rotation_aug = augmentation.RandomRotate90(dims=(0, 1), prob=1.0, max_number_rot=1)
        output_tensor = random_rotation_aug(self.random_tensor)
        self.assertFalse(torch.equal(self.random_tensor, output_tensor))

    def test_random_flip_equal(self):
        random_flip_aug = augmentation.RandomFlip(dims=(0,), prob=1.0)
        output_tensor = random_flip_aug(self.random_tensor)
        expected_output_tensor = torch.flip(self.random_tensor, (0,))
        self.assertTrue(torch.equal(expected_output_tensor, output_tensor))

    def test_random_flip_changed(self):
        random_flip_aug = augmentation.RandomFlip(dims=(0,), prob=1.0)
        output_tensor = random_flip_aug(self.random_tensor)
        self.assertFalse(torch.equal(self.random_tensor, output_tensor))


if __name__ == "__main__":
    unittest.main()

