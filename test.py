import unittest
import torch
from models import NetV3, NetV2, NetV4
import torch.nn as nn

# List of models that can be tested
TESTABLE_MODELS = {
    'NetV3': NetV3
    # Add other models here as needed
    # 'NetV2': NetV2,
    # 'NetV4': NetV4,
}

class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.model = NetV3()
        
    def test_parameter_count(self):
        """Test if model has less than 20k parameters"""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertLess(total_params, 20000, 
                       f"Model has {total_params} parameters, which exceeds the limit of 20,000")
        
    def test_batch_normalization_usage(self):
        """Test if model uses batch normalization"""
        has_batch_norm = any(isinstance(module, nn.BatchNorm2d) 
                           for module in self.model.modules())
        self.assertTrue(has_batch_norm, "Model should use BatchNormalization")
        
    def test_dropout_usage(self):
        """Test if model uses dropout"""
        has_dropout = any(isinstance(module, nn.Dropout) 
                         for module in self.model.modules())
        self.assertTrue(has_dropout, "Model should use Dropout")
        
    def test_gap_vs_fc(self):
        """Test if model uses Global Average Pooling instead of Fully Connected layers"""
        has_gap = any(isinstance(module, nn.AvgPool2d) 
                     for module in self.model.modules())
        has_fc = any(isinstance(module, nn.Linear) 
                    for module in self.model.modules())
        self.assertTrue(has_gap, "Model should use Global Average Pooling")
        self.assertFalse(has_fc, "Model should not use Fully Connected layers")

if __name__ == '__main__':
    unittest.main()
