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
                ("riya is my gf", "Samjha, riya tumhari girlfriend hai."),
                ("ha", "Haan, batao—kis baare mein baat karein?"),
                ("mera naam divu hai", "Theek hai, divu! Naam yaad rakhunga."),
                ("tumhara naam kya hai", "Mera naam Adivyanta hai."),
                ("i love you", "Yeh sunkar achha laga! Main Adivyanta hoon; tumse baat karne ke liye yahan hoon."),
                ("whats your name?", "Mera naam Adivyanta hai."),
                ("who is my gf?", "Tumne bataya tha ki riya tumhari girlfriend hai."),
                ("what is my name?", "Haan, tumhara naam divu hai."),
            ]
            with patch("smart_chat.generate", side_effect=AssertionError("unexpected model generation")):
                for user, expected in turns:
                    with self.subTest(user=user):
                        self.assertEqual(reply(None, None, memory, user), expected)
            self.assertEqual(Memory(memory.path).data["facts"]["name"], "divu")

    def test_second_user_transcript_and_consensual_roast(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(Path(directory) / "memory.json")
            turns = [
                ("hey riya meri gf ka naam hai", "Samjha, riya tumhari girlfriend hai."),
                ("tum riya ko jante ho?", "riya ke baare mein bas tumne bataya hai ki woh tumhari girlfriend hai. Main unhe personally nahi janta."),
                ("my name is divyanshu", "Theek hai, divyanshu! Naam yaad rakhunga."),
                ("tera naam kya hai?", "Mera naam Adivyanta hai."),
                ("gf kaise banaye?", "Pehle dosti aur genuine baat-cheet se shuru karo. Samne wale ki choice aur boundaries respect karo; connection dono taraf se ho tabhi aage badho."),
                ("acha beta", "Ji boss 😄 Ab batao, kya scene hai?"),
                ("adivyanta is chutiya", "Haan bhai, meri bakwaas pakad li. Aaj chutiya jawab diya ho toh seedha bol—dobara sahi se try karta hoon."),
                ("roast my code", "Tera code itna chutiya hai ki debugger bhi bolta hai: bhai, pehle tu saans le."),
                ("gaali de", "Saale, teri planning itni bakchodi hai ki kal bhi tujhe dekhkar postpone ho jata hai."),
            ]
            with patch("smart_chat.generate", side_effect=AssertionError("unexpected model generation")):
                for user, expected in turns:
                    with self.subTest(user=user):
                        self.assertEqual(reply(None, None, memory, user), expected)

    def test_followup_trip_transcript_uses_saved_task_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = Memory(path)
            with patch("smart_chat.generate", side_effect=AssertionError("unexpected model generation")):
                self.assertEqual(reply(None, None, memory, "meri gf riya hai"),
                                 "Samjha, riya tumhari girlfriend hai.")
                self.assertIn("batao kya hua", reply(None, None, memory, "im feeling low"))
                self.assertIn("small language model", reply(None, None, memory, "are you an slm?"))
                self.assertIn("arithmetic", reply(None, None, memory, "acha kya kar sakte ho"))
                self.assertIn("2-day starter plan", reply(None, None, memory, "trip panchmadi ki hai"))
                memory = Memory(path)  # Task context survives a chat restart.
                plan = reply(None, None, memory, "yes make it")
                self.assertIn("Pandav Caves", plan)
                self.assertIn("Bee Falls", plan)
                self.assertIn("mptourism.com/destination-panchmarhi.php", plan)
                self.assertEqual(reply(None, None, memory, "yes make it"),
                                 "Kya plan banaun? Destination aur kitne din ke liye, dono bata do.")


if __name__ == "__main__":
    unittest.main()
