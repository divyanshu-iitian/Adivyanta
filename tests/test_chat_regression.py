"""The transcript that exposed simple app-routing failures."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from conversation_memory import Memory
from smart_chat import reply


class ChatTranscriptTests(unittest.TestCase):
    def test_known_turns_never_fall_through_to_neural_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(Path(directory) / "memory.json")
            turns = [
                ("hey", "Namaste! Aaj kya baat karein?"),
                ("mera naam divyanshu", "Theek hai, divyanshu! Naam yaad rakhunga."),
                ("5+5 kya hota hai", "10"),
                ("adita is my gf", "Samjha, adita tumhari girlfriend hai."),
                ("ha", "Haan, batao—kis baare mein baat karein?"),
                ("mera naam divu hai", "Theek hai, divu! Naam yaad rakhunga."),
                ("tumhara naam kya hai", "Mera naam Adivyanta hai."),
                ("i love you", "Yeh sunkar achha laga! Main Adivyanta hoon; tumse baat karne ke liye yahan hoon."),
                ("whats your name?", "Mera naam Adivyanta hai."),
                ("who is my gf?", "Tumne bataya tha ki adita tumhari girlfriend hai."),
                ("what is my name?", "Haan, tumhara naam divu hai."),
            ]
            with patch("smart_chat.generate", side_effect=AssertionError("unexpected model generation")):
                for user, expected in turns:
                    with self.subTest(user=user):
                        self.assertEqual(reply(None, None, memory, user), expected)
            self.assertEqual(Memory(memory.path).data["facts"]["name"], "divu")


if __name__ == "__main__":
    unittest.main()
