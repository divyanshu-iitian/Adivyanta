"""Small, inspectable local memory for the interactive Adivyanta app."""
import json
import os
from pathlib import Path
import re

from travel_planner import mentioned_trip


DEFAULT_PATH = Path(__file__).parent / "data" / "adivyanta_memory.json"
MAX_NOTES = 50
MAX_TURNS = 40


class Memory:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"facts": {}, "notes": [], "likes": [], "turns": [], "corrections": {}, "state": {}}
        for key, default in (("facts", {}), ("notes", []), ("likes", []), ("turns", []), ("corrections", {}), ("state", {})):
            self.data.setdefault(key, default)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, self.path)

    def learn(self, message):
        """Extract a few explicit facts; never treat arbitrary chat as a true fact."""
        changes = []
        name = re.search(r"\b(?:my name is|call me|mera naam|मेरा नाम)\s+([^.!?,।]{2,35})", message, re.I)
        if name:
            value = name.group(1).strip(" .,!?")
            value = re.split(r"\b(?:and|aur|hai|hoon|h[u]?n|please|bro)\b", value, maxsplit=1, flags=re.I)[0].strip()
            value = re.split(r"\s+(?:है|हूँ)$", value, maxsplit=1)[0].strip()
            if value and len(value.split()) <= 3:
                self.data["facts"]["name"] = value
                changes.append("name")
        girlfriend = re.fullmatch(
            r"\s*([A-Za-z][A-Za-z -]{1,34})\s+is my\s+(?:gf|girlfriend)\s*[.!]?\s*",
            message, re.I,
        )
        if not girlfriend:
            girlfriend = re.search(
                r"\b([A-Za-z]{2,25})\s+meri\s+(?:gf|girlfriend)\s+(?:ka\s+naam\s+)?hai\b",
                message, re.I,
            )
        if not girlfriend:
            girlfriend = re.search(
                r"\bmeri\s+(?:gf|girlfriend)\s+ka\s+naam\s+(?:hai\s+)?([A-Za-z]{2,25})\b",
                message, re.I,
            )
        if not girlfriend:
            girlfriend = re.search(r"\bmeri\s+(?:gf|girlfriend)\s+([A-Za-z]{2,25})\s+hai\b", message, re.I)
        if girlfriend:
            self.data["facts"]["girlfriend"] = girlfriend.group(1).strip()
            changes.append("girlfriend")
        if mentioned_trip(message):
            self.data["state"].update({"pending_task": "pachmarhi_plan",
                                       "pending_task_turn": len(self.data["turns"])})
            changes.append("trip")
        language = re.search(r"\b(?:speak|talk|reply|respond|i prefer)\s+(?:to me\s+)?(?:in\s+)?(hindi|hinglish|english)\b|\b(hindi|hinglish|english)\s+(?:mein|me)\s+(?:baat|reply|bolo|bolna)\b", message, re.I)
        if language:
            self.data["facts"]["language"] = (language.group(1) or language.group(2)).lower()
            changes.append("language")
        if re.search(r"\b(?:you are made by me|i (?:built|created|made) you|maine tumhe banaya)\b", message, re.I):
            self.data["facts"]["creator"] = "user"
            changes.append("creator")
        note = re.search(r"\b(?:remember that|yaad rakhna ki)\s+(.+)", message, re.I)
        if note:
            self.remember(note.group(1).strip(" .!?"))
            changes.append("note")
        like = re.search(r"\b(?:i like|i love)\s+(.{2,60}?)[.!?]?$|\bmujhe\s+(.{2,60}?)\s+pasand hai[.!?]?$",
                         note.group(1) if note else message, re.I)
        if like and "?" not in message:
            item = (like.group(1) or like.group(2)).strip(" .!?")
            if item and item.casefold() not in {"you", "u", "tum", "tumhe", "aap"} and item.casefold() not in {x.casefold() for x in self.data["likes"]}:
                self.data["likes"].append(item)
                self.data["likes"] = self.data["likes"][-20:]
                changes.append("likes")
        if changes:
            self.save()
        return changes

    def remember(self, note):
        note = note.strip()
        if not note or len(note) > 300:
            raise ValueError("Memory note must be 1–300 characters.")
        if note not in self.data["notes"]:
            self.data["notes"].append(note)
            self.data["notes"] = self.data["notes"][-MAX_NOTES:]
            self.save()

    def add_turn(self, role, content):
        self.data["turns"].append({"role": role, "content": content})
        self.data["turns"] = self.data["turns"][-MAX_TURNS:]
        self.save()

    def facts_text(self):
        facts = self.data["facts"]
        lines = []
        if facts.get("name"):
            lines.append(f"The user's name is {facts['name']}.")
        if facts.get("language"):
            lines.append(f"The user prefers {facts['language']}.")
        if facts.get("creator") == "user":
            lines.append("The user says they built this Adivyanta project. Acknowledge this without claiming consciousness.")
        if facts.get("girlfriend"):
            lines.append(f"The user says their girlfriend's name is {facts['girlfriend']}.")
        lines.extend(f"User asked you to remember: {note}" for note in self.data["notes"][-20:])
        if self.data["likes"]:
            lines.append("The user likes: " + ", ".join(self.data["likes"]) + ".")
        return "\n".join(lines)

    def known_answer(self, question):
        q = question.lower().strip(" ?!.")
        name = self.data["facts"].get("name")
        if name and (re.search(r"\b(?:what(?:'s| is) my name|do you know my name|remember my name|mera naam kya|mera naam yaad)\b", q)
                     or "मेरा नाम क्या" in q
                     or q in {"my name", "mera naam"}):
            return f"तुम्हारा नाम {name} है।" if "मेरा नाम" in q else f"Haan, tumhara naam {name} hai."
        if self.data["likes"] and re.search(r"\b(?:what do i like|what food do i like|what do you know i like|mujhe kya pasand)\b", q):
            return "Tumhe " + ", ".join(self.data["likes"]) + " pasand hai."
        girlfriend = self.data["facts"].get("girlfriend")
        if girlfriend and re.search(r"\b(?:what(?:'s| is) my (?:gf|girlfriend)(?:'s name)?|who is my (?:gf|girlfriend)|meri (?:gf|girlfriend) ka naam kya|meri (?:gf|girlfriend) kaun)\b", q):
            return f"Tumne bataya tha ki {girlfriend} tumhari girlfriend hai."
        person_query = re.search(r"\b(?:tum|kya tum)\s+([a-z]{2,25})\s+ko\s+(?:jante|jaante)\s+ho$|\bdo you know\s+([a-z]{2,25})$", q)
        if person_query:
            person = person_query.group(1) or person_query.group(2)
            if girlfriend and person.casefold() == girlfriend.casefold():
                return f"{girlfriend} ke baare mein bas tumne bataya hai ki woh tumhari girlfriend hai. Main unhe personally nahi janta."
            return f"{person} ke baare mein tumne mujhe abhi kuch nahi bataya."
        return self.data["corrections"].get(q)

    def correct(self, question, answer):
        question, answer = question.strip().lower().strip(" ?!."), answer.strip()
        if not question or not answer or len(question) > 300 or len(answer) > 500:
            raise ValueError("Use /correct question => better answer (up to 300/500 characters).")
        self.data["corrections"][question] = answer
        self.save()

    def forget(self, key):
        key = key.strip().lower()
        if key == "all":
            self.data = {"facts": {}, "notes": [], "likes": [], "turns": [], "corrections": {}, "state": {}}
        elif key in self.data["facts"]:
            del self.data["facts"][key]
        elif key == "notes":
            self.data["notes"] = []
        elif key == "likes":
            self.data["likes"] = []
        elif key == "history":
            self.data["turns"] = []
            self.data["state"].pop("pending_task", None)
            self.data["state"].pop("pending_task_turn", None)
        elif key == "trip":
            self.data["state"].pop("pending_task", None)
            self.data["state"].pop("pending_task_turn", None)
        else:
            return False
        self.save()
        return True
