import unittest

from grounded_tools import arithmetic_answer


class GroundedToolsTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(arithmetic_answer("what is 23 * (4 + 2)?"), "138")
        self.assertEqual(arithmetic_answer("7 / 2"), "7/2")

    def test_refuses_other_text_and_unsafe_expressions(self):
        self.assertIsNone(arithmetic_answer("I have 2 keys and 3 bags"))
        self.assertEqual(arithmetic_answer("2 ** 100000"), "I can't calculate that expression.")
        self.assertEqual(arithmetic_answer("2 / 0"), "I can't calculate that expression.")
        self.assertIsNone(arithmetic_answer("__import__('os')"))
