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

    def test_relationship_fact_and_affection_do_not_pollute_likes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            first = Memory(path)
            self.assertIn("girlfriend", first.learn("riya is my gf"))
            first.learn("i love you")
            second = Memory(path)
            self.assertEqual(second.known_answer("who is my gf?"), "Tumne bataya tha ki riya tumhari girlfriend hai.")
            self.assertEqual(second.data["likes"], [])
            self.assertTrue(second.forget("girlfriend"))
            self.assertIsNone(Memory(path).known_answer("who is my gf?"))

    def test_hinglish_relationship_and_acquaintance(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(Path(directory) / "memory.json")
            self.assertIn("girlfriend", memory.learn("hey riya meri gf ka naam hai"))
            self.assertEqual(memory.known_answer("tum riya ko jante ho?"),
                             "riya ke baare mein bas tumne bataya hai ki woh tumhari girlfriend hai. Main unhe personally nahi janta.")
            self.assertEqual(memory.known_answer("tum neha ko jante ho?"),
                             "neha ke baare mein tumne mujhe abhi kuch nahi bataya.")


if __name__ == "__main__":
    unittest.main()
