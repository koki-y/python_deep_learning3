import unittest

from variable import *

from dezero.core_simple import *

class AddTest(unittest.TestCase):
    def test_forward(self):
        y = add(Variable(np.array(3)), Variable(np.array(4)))
        expected = np.array(7)
        self.assertEqual(y.data, expected)

    def test_same_var(self):
        x = Variable(np.array(3))
        y = add(x, x)
        y.backward(retain_grad=True)

        self.assertEqual(y.data, np.array(6))
        self.assertEqual(y.grad, np.array(1))
        self.assertEqual(x.grad, np.array(2))

class MulTest(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        y = add(mul(a, b), c)
        y.backward()

        self.assertEqual(y.data, np.array(7.0))
        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 3.0)
        self.assertEqual(c.grad, 1.0)

class NegTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = -x
        y.backward()

        self.assertEqual(y.data, np.array(-2.0))
        self.assertEqual(x.grad, -1.0)

class SubTest(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array(2.0))
        b = Variable(np.array(3.0))
        y = sub(a, b)
        y.backward()

        self.assertEqual(y.data, np.array(-1.0))
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, -1.0)

class DivTest(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array(4.0))
        b = Variable(np.array(2.0))
        y = div(a, b)
        y.backward()

        self.assertEqual(y.data, np.array(2.0))
        self.assertEqual(a.grad, 0.5)
        self.assertEqual(b.grad, -1.0)

class powTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = pow(x, 4)
        y.backward()

        self.assertEqual(y.data, np.array(16.0))
        self.assertEqual(x.grad, 32.0)

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

    def test_huge_calc(self):
        for i in range(10):
            x = Variable(np.random.randn(10000))
            y = square(square(square(x)))

class CombinationTest(unittest.TestCase):
    def test_squared_add(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))

        z = add(square(x), square(y))
        z.backward(retain_grad=True)

        self.assertEqual(z.data, np.array(13))
        self.assertEqual(x.grad, np.array(4))
        self.assertEqual(y.grad, np.array(6))
    
    def test_complex_calc_model(self):
        x = Variable(np.array(2))

        a = square(x)
        b = square(a)
        c = square(a)
        y = add(b, c)
        y.backward(retain_grad=True)

        self.assertEqual(y.data, np.array(32))
        self.assertEqual(y.grad, np.array(1))
        self.assertEqual(c.grad, np.array(1))
        self.assertEqual(b.grad, np.array(1))
        self.assertEqual(a.grad, np.array(16))
        self.assertEqual(x.grad, np.array(64))

    def test_retain_grad(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward(retain_grad=False)

        self.assertEqual(y.grad, None)
        self.assertEqual(t.grad, None)
        self.assertEqual(x0.grad, np.array(2.0))
        self.assertEqual(x1.grad, np.array(1.0))

    def test_no_grad(self):
        with no_grad():
            x = Variable(np.array(2.0))
            y = square(x)

        self.assertEqual(y.generation, 0)
        self.assertFalse(hasattr(y, "inputs"))
        self.assertFalse(hasattr(y, "outputs"))

class OperatorTest(unittest.TestCase):
    def test_add_and_mul(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        y = a * b + c
        y.backward()

        self.assertEqual(y.data, np.array(7.0))
        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 3.0)
        self.assertEqual(c.grad, 1.0)

    def test_with_np_array(self):
        x = Variable(np.array(2.0))
        y = x + np.array(3.0)

        self.assertEqual(y.data, np.array(5.0))

    def test_for_left_value(self):
        x = Variable(np.array(2.0))
        y = 2.0 * (1.0 + np.array(2.0) * (np.array(3.0) + x))

        self.assertEqual(y.data, np.array(22.0))

    def test_neg(self):
        x = Variable(np.array(2.0))
        y = -x
        y.backward()

        self.assertEqual(y.data, np.array(-2.0))
        self.assertEqual(x.grad, -1.0)

    def test_sub(self):
        a = Variable(np.array(2.0))
        y = 10.0 - np.array(1.0) - a - np.array(3.0)
        y.backward()

        self.assertEqual(y.data, np.array(4.0))
        self.assertEqual(a.grad, -1.0)

    def test_div(self):
        x = Variable(np.array(4.0))
        y = x / 2.0
        y.backward()

        self.assertEqual(y.data, np.array(2.0))
        self.assertEqual(x.grad, 0.5)

    def test_pow(self):
        x = Variable(np.array(2.0))
        y = x ** 4.0
        y.backward()

        self.assertEqual(y.data, np.array(16.0))
        self.assertEqual(x.grad, 32.0)

