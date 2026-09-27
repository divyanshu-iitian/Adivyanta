"""Small source-backed offline trip plan for the destination in the user transcript."""
import re


SOURCE_URL = "https://www.mptourism.com/destination-panchmarhi.php"
PACHMARHI_RE = re.compile(r"\b(?:pachmarhi|panchmarhi|panchmadi)\b", re.I)
TRIP_RE = re.compile(r"\b(?:trip|tour|travel|visit|ghumne|ghoomne|itinerary|plan)\b", re.I)
MAKE_RE = re.compile(r"^(?:yes(?:,?\s+(?:please\s+)?)?(?:make it|make one|plan it)?|haan(?:\s+(?:bana do|plan bana do))?|plan bana do|make it|make one|bana do)[.!?\s]*$", re.I)


def mentioned_trip(message):
    return bool(PACHMARHI_RE.search(message) and TRIP_RE.search(message))


def asks_to_make_plan(message):
    return bool(MAKE_RE.fullmatch(message.strip()))


def pachmarhi_plan():
    return (
        "Pachmarhi ka starter 2-day plan (dates aur transport abhi unknown):\n"
        "Day 1: Pandav Caves aur Jata Shankar; shaam ko Dhoopgarh, agar us din access aur weather theek ho.\n"
        "Day 2: Bee Falls aur Apsara Vihar ke liye time rakho; chalne aur local transport ka margin rakho.\n"
        "Nikalne se pehle entry, route aur weather locally verify kar lena. "
        f"Places: MP Tourism {SOURCE_URL}\n"
        "Kitne din aur kis city se jaoge? Bataoge toh plan adjust kar dunga."
    )
