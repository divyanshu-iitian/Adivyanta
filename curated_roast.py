"""Transparent authored fallback for common, consent-based roast topics."""
import re

from roast_data import HINGLISH, TOPICS


ALIASES = [
    ("my buggy code", r"\b(code|coding|bug|debug|compile|program|script)\b"),
    ("my coding comments", r"\b(comment|documentation|todo)\b"),
    ("my 47 browser tabs", r"\b(browser|tabs?)\b"),
    ("my procrastination", r"\b(procrastinat\w*|delay|deadline)\b"),
    ("my skipped workouts", r"\b(gym|workout|fitness|exercise)\b"),
    ("my sleep schedule", r"\b(sleep|bedtime|alarm)\b"),
    ("my exam preparation", r"\b(exam|study|studying|padhai|syllabus)\b"),
    ("my coffee habit", r"\b(coffee|espresso|cafe)\b"),
    ("my gaming aim", r"\b(gaming|gamer|aim|crosshair|shooting)\b"),
    ("my phone battery", r"\b(phone|battery|charger)\b"),
    ("my to-do list", r"\b(to-do|todo|tasks?|checklist)\b"),
    ("my meeting habits", r"\b(meeting|sync|standup)\b"),
    ("my overflowing inbox", r"\b(inbox|emails?)\b"),
    ("my cooking", r"\b(cook|cooking|recipe|noodles|pasta)\b"),
    ("my photography", r"\b(photo|camera|picture|photography)\b"),
    ("my music playlist", r"\b(playlist|music|songs?)\b"),
    ("my outfit choices", r"\b(outfit|wardrobe|clothes|fashion)\b"),
    ("my dancing", r"\b(dance|dancing)\b"),
    ("my presentation slides", r"\b(slides?|presentation|deck)\b"),
    ("my startup pitch", r"\b(startup|pitch|business model)\b"),
    ("my social media posts", r"\b(social media|hashtags?|posts?|drafts?)\b"),
    ("my AI prompts", r"\b(ai prompt|prompting|prompts?)\b"),
    ("my messy desk", r"\b(desk|sticky notes)\b"),
    ("my travel planning", r"\b(travel|vacation\w*|itinerary|trip|spreadsheet\w*)\b"),
    ("my cricket batting", r"\b(cricket|batting|cover drive)\b"),
    ("my chess openings", r"\b(chess|queen|gambit)\b"),
    ("my video editing", r"\b(video editing|timeline|jump cuts)\b"),
    ("my job interview answers", r"\b(interview)\b"),
    ("my handwriting", r"\b(handwriting|handwritten)\b"),
    ("my online shopping cart", r"\b(shopping|cart|wishlist)\b"),
]

ENGLISH = dict(TOPICS)
ENGLISH["my unused notebooks"] = ["Your notebooks are so untouched they still think the shop is home."]
ROMAN_HINDI = dict(HINGLISH)
HINGLISH_MAP = {
    "my buggy code": "meri coding",
    "my 47 browser tabs": "mere browser tabs",
    "my procrastination": "meri procrastination",
    "my exam preparation": "meri padhai",
    "my coffee habit": "meri coffee habit",
    "my gaming aim": "mera gaming aim",
}
ALIASES.append(("my unused notebooks", r"\b(notebook\w*|journal\w*)\b"))


def choose(prompt):
    """Return a human-authored line and matched topic, or (None, None)."""
    p = prompt.lower()
    hinglish = bool(re.search(r"\b(meri|mere|mera|karo|hinglish|padhai)\b", p))
    for topic, pattern in ALIASES:
        if re.search(pattern, p):
            if hinglish and topic in HINGLISH_MAP:
                return ROMAN_HINDI[HINGLISH_MAP[topic]][0], topic
            return ENGLISH[topic][0], topic
    return None, None
