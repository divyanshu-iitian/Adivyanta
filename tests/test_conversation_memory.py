import tempfile
import unittest
from pathlib import Path

from conversation_memory import Memory


class MemoryTests(unittest.TestCase):
    def test_name_correction_and_forget_persist_across_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            first = Memory(path)
            first.learn("my name is Divyanshu")
            first.correct("kya kar rahe ho", "Tumse baat kar raha hoon.")
            second = Memory(path)
            self.assertEqual(second.known_answer("do you know my name?"), "Haan, tumhara naam Divyanshu hai.")
            self.assertEqual(second.known_answer("kya kar rahe ho?"), "Tumse baat kar raha hoon.")
            self.assertTrue(second.forget("all"))
            self.assertIsNone(Memory(path).known_answer("do you know my name?"))

    def test_explicit_note_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            Memory(path).learn("remember that I like samosas")
            self.assertIn("I like samosas", Memory(path).facts_text())
            self.assertEqual(Memory(path).known_answer("what do I like?"), "Tumhe samosas pasand hai.")

    def test_devanagari_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            Memory(path).learn("मेरा नाम दिव्यांशु है।")
            self.assertEqual(Memory(path).known_answer("मेरा नाम क्या है?"), "तुम्हारा नाम दिव्यांशु है।")


if __name__ == "__main__":
    unittest.main()
