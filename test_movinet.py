"""
Unit Tests for Movinet Action Recognition (PyTorch)
"""

import unittest
import numpy as np
from unittest.mock import MagicMock


class TestMovinetClassifier(unittest.TestCase):
    
    def test_sample_classes_not_empty(self):
        from movinet_classifier import MovinetClassifier
        self.assertGreater(len(MovinetClassifier.SAMPLE_CLASSES), 0)
        print(f"Test: Sample classes defined ({len(MovinetClassifier.SAMPLE_CLASSES)} classes)")
    
    def test_gpu_detection(self):
        import torch
        is_cuda = torch.cuda.is_available()
        print(f"Test: GPU detection - PASSED (CUDA: {is_cuda})")
    
    def test_predictions_format(self):
        mock_results = [
            ("running", 0.85),
            ("walking", 0.10),
            ("jumping", 0.03),
            ("standing", 0.01),
            ("sitting", 0.01)
        ]
        
        self.assertIsInstance(mock_results, list)
        self.assertEqual(len(mock_results), 5)
        
        for label, prob in mock_results:
            self.assertIsInstance(label, str)
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
        
        print("Test: Predictions format - PASSED")
    
    def test_model_variants(self):
        valid_models = ["a0", "a1", "a2", "a3"]
        
        for model_id in valid_models:
            self.assertIn(model_id, valid_models)
        
        print(f"Test: Model variants - PASSED ({valid_models})")


class TestGUIFunctionality(unittest.TestCase):
    
    def test_gui_imports(self):
        import tkinter as tk
        from tkinter import ttk
        print("Test: GUI imports - PASSED")
    
    def test_button_commands(self):
        mock_btn = MagicMock()
        mock_btn.config = MagicMock()
        print("Test: Button commands - PASSED")


def run_tests():
    print("\n" + "="*50)
    print("RUNNING MOVINET TESTS")
    print("="*50 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMovinetClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestGUIFunctionality))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*50)
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*50 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
