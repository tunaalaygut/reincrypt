import unittest
import sys
sys.path.append('..')
from src.utility.r_utils import bound_scalar


class BoundScalarUnitTest(unittest.TestCase):
    def test_bound_scalar_upper_bound(self):
        self.assertEqual(20, bound_scalar(43))
    
    def test_bound_scalar_lower_bound(self):
        self.assertEqual(-20, bound_scalar(-123))
    
    def test_bound_scalar(self):
        self.assertEqual(11, bound_scalar(11))


if __name__ == "__main__":
    unittest.main()
