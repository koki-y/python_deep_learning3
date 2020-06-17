import unittest

from variable import *

class AddTest(unittest.TestCase):
    def test_forward(self):
        y = add(Variable(np.array(3)), Variable(np.array(4)))
        expected = np.array(7)
        self.assertEqual(y.data, expected)
    
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

class CombinationTest(unittest.TestCase):
    def test_squared_add(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))

        z = add(square(x), square(y))
        z.backward()

        self.assertEqual(z.data, np.array(13))
        self.assertEqual(x.grad, np.array(4))
        self.assertEqual(y.grad, np.array(6))

