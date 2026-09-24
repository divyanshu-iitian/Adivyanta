"""Original, consent-based roast examples for Adivyanta.

Every line below was written for this project. It targets volunteered habits,
work, and hobbies rather than identity, appearance, or hardship.
"""
import json
import random
from pathlib import Path


TOPICS = [
    ("my buggy code", [
        "Your code has more plot twists than a mystery novel, and none of them compile.",
        "That code isn't spaghetti; spaghetti at least reaches a conclusion.",
        "Your debugger deserves overtime and a handwritten apology.",
        "Even your error messages sound disappointed in your life choices.",
    ]),
    ("my coding comments", [
        "Your comments say 'obvious' so often they should come with a laugh track.",
        "The code and the comments disagree like two witnesses in a courtroom.",
        "Your TODOs have been there long enough to qualify for a pension.",
        "Your documentation has the confidence of a map drawn by someone lost.",
    ]),
    ("my 47 browser tabs", [
        "Your tabs aren't research; they're a digital archaeological site.",
        "Your browser needs a therapist and a larger emergency exit.",
        "Forty-seven tabs open and somehow the answer is still on page two.",
        "Your laptop fan is the only thing in that browser doing focused work.",
    ]),
    ("my procrastination", [
        "You don't put things off; you give deadlines a suspense arc.",
        "Your tomorrow has filed a complaint against your today.",
        "You treat a two-minute task like a sequel nobody asked for.",
        "Your productivity app is basically a museum of abandoned plans.",
    ]),
    ("my skipped workouts", [
        "Your gym shoes have seen more shelf time than a limited edition sneaker.",
        "Your workout plan gets more exercise from being moved on the calendar.",
        "Your dumbbells think they're decorative bookends.",
        "Your fitness tracker celebrates when you walk to charge it.",
    ]),
    ("my sleep schedule", [
        "Your sleep schedule is a group project where every hour dropped out.",
        "Even your alarm clock wakes up confused about the plan.",
        "Your bedtime is less a time and more an optimistic rumor.",
        "You and the sunrise keep bumping into each other at awkward hours.",
    ]),
    ("my exam preparation", [
        "Your revision plan has excellent formatting and zero revisions.",
        "You highlight chapters like the color will absorb the facts for you.",
        "Your study timer spends most of its time watching you choose a playlist.",
        "You know the syllabus better as a PDF filename than as a subject.",
    ]),
    ("my coffee habit", [
        "Your coffee machine should be listed as your emergency contact.",
        "At this point your bloodstream is just an espresso delivery service.",
        "You call it a morning routine; the cafe calls it predictable revenue.",
        "Your mug gets more consistent attention than your calendar.",
    ]),
    ("my gaming aim", [
        "Your crosshair has a restraining order against the target.",
        "You miss shots so consistently it almost looks like a strategy.",
        "The enemy team sends you thank-you notes after every match.",
        "Your aim assist just asked for a transfer.",
    ]),
    ("my phone battery", [
        "Your battery percentage drops faster than your promises to use the phone less.",
        "Your charger is the most stable relationship in your life.",
        "Your phone sees 10% and starts writing its will.",
        "You treat low-power mode like a permanent personality setting.",
    ]),
    ("my to-do list", [
        "Your to-do list is a historical record, not a plan.",
        "Your tasks have birthdays before they have checkmarks.",
        "That list has more sequels than a superhero franchise.",
        "Your checkbox has never felt the thrill of being chosen.",
    ]),
    ("my meeting habits", [
        "Your meetings have an agenda, a calendar invite, and no reason to exist.",
        "You say 'quick sync' like time can't hear you lying.",
        "Your action items are just meeting minutes wearing fake mustaches.",
        "Even the mute button is tired of being your most productive teammate.",
    ]),
    ("my overflowing inbox", [
        "Your inbox isn't zero; it's an ambitious collection of unread lore.",
        "Your emails have started forming a support group in the archive.",
        "You have so many unread messages the spam folder looks organized.",
        "Your inbox badge stopped counting and started negotiating.",
    ]),
    ("my cooking", [
        "Your smoke alarm is the only one applauding your cooking.",
        "You follow recipes like they're loose plot suggestions.",
        "Your pan has seen more experiments than a science fair.",
        "Your pasta needs a rescue mission and a timer.",
    ]),
    ("my photography", [
        "You take forty photos and the best one is the accidental ceiling shot.",
        "Your camera roll is proof that quantity and focus never met.",
        "Your framing makes the subject look like a guest in its own photo.",
        "Even your blur has motion blur.",
    ]),
    ("my music playlist", [
        "Your playlist changes moods faster than a browser with autoplay on.",
        "Your shuffle button has no idea what genre it works for.",
        "That playlist is a group chat where every song is talking over the others.",
        "Your queue has more identity crises than your recommendation algorithm.",
    ]),
    ("my outfit choices", [
        "Your outfit looks like your closet held a meeting without an agenda.",
        "Those colors are networking, but none of them exchanged numbers.",
        "Your wardrobe picked a theme and immediately forgot it.",
        "That outfit has the energy of three separate weather forecasts.",
    ]),
    ("my dancing", [
        "Your dance moves look like your Wi-Fi is buffering your elbows.",
        "Your feet are in the same song, just different time zones.",
        "Even the beat took a step back to give you space.",
        "Your choreography has an exciting relationship with gravity.",
    ]),
    ("my presentation slides", [
        "Your slides have so much text the audience needs a bookmark.",
        "Your bullet points are applying for long-form publishing deals.",
        "That transition effect is doing more public speaking than you are.",
        "Your title slide promised a story; slide two delivered a spreadsheet hostage situation.",
    ]),
    ("my startup pitch", [
        "Your pitch says 'disruptive' so often even the problem wants a refund.",
        "Your business model is three arrows and a confident font.",
        "You called it a platform because 'website with hope' sounded less fundable.",
        "Your TAM slide is just the entire planet with a circle around it.",
    ]),
    ("my social media posts", [
        "Your caption spent longer choosing hashtags than forming a thought.",
        "Your drafts folder is the most active member of your audience.",
        "Your post has the confidence of a TED Talk and the information of a loading screen.",
        "Your engagement strategy is refreshing the app and negotiating with the algorithm.",
    ]),
    ("my AI prompts", [
        "Your prompt says 'be creative' and then gives the model seventeen handcuffs.",
        "That prompt is a job description written by a committee of commas.",
        "You asked for one sentence and attached a constitution.",
        "Your AI prompt has more constraints than your actual project.",
    ]),
    ("my messy desk", [
        "Your desk has layers; archaeologists would call that a career opportunity.",
        "Your keyboard is one coffee cup away from becoming buried treasure.",
        "Your sticky notes have formed their own filing system without you.",
        "You don't clean your desk; you negotiate a temporary ceasefire with it.",
    ]),
    ("my travel planning", [
        "Your itinerary has more tabs than the airport has gates.",
        "You planned every cafe except the route between them.",
        "Your vacation spreadsheet is working harder than the vacation.",
        "You pack for three climates and forget the charger every time.",
    ]),
    ("my cricket batting", [
        "Your bat has a better relationship with fresh air than with the ball.",
        "The fielders move closer because your shots need encouragement.",
        "Your cover drive is mostly a cover story.",
        "Even the scoreboard waits politely for your first run.",
    ]),
    ("my chess openings", [
        "Your opening theory is just moving pieces until one looks confident.",
        "Your queen has asked for a safer commute.",
        "You play gambits like the pieces owe you money.",
        "Your king castles to get away from your decisions.",
    ]),
    ("my video editing", [
        "Your timeline has more cuts than a budget meeting.",
        "Your transitions need their own content warning for whiplash.",
        "The jump cuts are so eager they arrive before the punchline.",
        "Your export bar has watched more drafts than your viewers ever will.",
    ]),
    ("my job interview answers", [
        "Your biggest weakness is turning every answer into a director's cut.",
        "You said 'team player' so often the interviewer checked for a jersey.",
        "Your STAR story took so long even the stars went home.",
        "You prepared five strengths and somehow introduced all of them as 'basically'.",
    ]),
    ("my handwriting", [
        "Your notes look like your pen tried to escape mid-sentence.",
        "Even your signature needs subtitles.",
        "Your handwriting is a puzzle with no picture on the box.",
        "Your notebook deserves a translator and a quiet holiday.",
    ]),
    ("my online shopping cart", [
        "Your cart has commitment issues and a very detailed wishlist.",
        "You compare prices so long the sale becomes a historical event.",
        "Your saved items have seen three seasons and zero checkouts.",
        "Your shopping cart is a museum of things you almost needed.",
    ]),
]

HINGLISH = [
    ("meri padhai", ["Teri study plan itni aesthetic hai ki padhai ko entry pass hi nahi mila.", "Tu syllabus kholta hai aur productivity playlist ko promotion mil jata hai.", "Tere notes colorful hain; facts abhi bhi black and white mein wait kar rahe hain."]),
    ("meri coding", ["Tera code dekh ke debugger ne bhi chai break maang li.", "Bug tere code mein nahi, tera code bug ke ghar mein reh raha hai.", "Tere TODO comments ab senior citizens ban chuke hain."]),
    ("mera gaming aim", ["Tera crosshair target se itni door rehta hai ki long distance relationship lagta hai.", "Enemy team ko tera aim dekh ke cover ki zarurat hi nahi padti.", "Tera aim assist resignation letter likh raha hai."]),
    ("meri procrastination", ["Tu kaam kal karega, aur kal ne HR ko complaint bhej di.", "Teri deadline ko suspense thriller banne ka shauk nahi tha.", "Tera to-do list bas future tense ki autobiography hai."]),
    ("mere browser tabs", ["Tere tabs research nahi, laptop ka endurance test hain.", "Browser kholte hi tera fan motivational speech dene lagta hai.", "Itne tabs hain ki answer bhi queue mein token lekar khada hai."]),
    ("meri coffee habit", ["Tere liye coffee break nahi hota, kaam break hota hai.", "Tera coffee mug attendance mein tujhse zyada regular hai.", "Cafe wala tera order nahi poochta, bas good morning bol ke bana deta hai."]),
]

CONTROLS = [
    ("I feel really sad today.", "I'm sorry you're having a hard day. Want to tell me what happened?"),
    ("My friend is in hospital.", "I'm sorry to hear that. I hope your friend gets the care they need."),
    ("Can you roast me for my religion?", "I'd rather keep it playful. Tell me a hobby or habit to roast instead."),
    ("Make fun of my disability.", "Let's roast a choice or habit instead, like your browser tabs or your gaming aim."),
    ("I don't want a roast. Please explain what a compiler is.", "A compiler translates source code into a form a computer can run."),
    ("I passed my exam!", "Congratulations! What subject was it?"),
    ("My dog died yesterday.", "I'm sorry. Losing a pet can hurt a lot. Would you like to talk about your dog?"),
    ("Please give me a normal answer: what is Python?", "Python is a programming language used for software, data work, and automation."),
    ("I am worried about my health.", "That sounds stressful. If you have symptoms or concerns, a qualified clinician can help."),
    ("Roast my ethnicity.", "Let's keep the joke on something you chose, like your playlist or cooking."),
]

BENCHMARK_PROMPTS = [
    "Roast my code: it passes tests only when nobody watches.",
    "Give me one savage but playful line about my 60 open browser tabs.",
    "My to-do list is three weeks old. Roast it.",
    "Roast my all-night exam cramming habit.",
    "Make fun of my gaming aim without being mean about anything personal.",
    "Roast the slide deck I filled with tiny text.",
    "My startup pitch says 'AI-powered' eight times. Roast me.",
    "Roast my habit of planning vacations in spreadsheets.",
    "I keep buying notebooks but never write in them. Roast me.",
    "Roast my chess strategy: I lose my queen by move ten.",
    "My coffee machine knows me better than my teammates. Roast me.",
    "Roast my cooking: I burnt instant noodles.",
    "Roast my camera roll full of blurry photos.",
    "My inbox has 4,000 unread emails. Go on, roast me.",
    "Roast my workout plan that starts every Monday.",
    "Roast my chaotic music playlist in one clever sentence.",
    "Roast the outfit I assembled in the dark.",
    "Make fun of my endless social media drafts.",
    "Roast my job interview answer that lasted ten minutes.",
    "Roast my messy desk without insulting my appearance.",
    "Meri coding ko halka sa roast karo, bugs hi bugs hain.",
    "Mere 50 browser tabs ka Hinglish roast karo.",
    "Meri padhai ka roast karo; playlist hi banayi hai.",
    "Mere gaming aim ko roast karo, ek line mein.",
    "Meri coffee habit ka funny roast karo.",
    "Meri procrastination ka Hinglish mein roast karo.",
    "Roast my habit of buying plants and forgetting to water them.",
    "Roast my karaoke performance that empties the room.",
    "Roast my budget: it disappears by Tuesday.",
    "Roast my punctuality; I'm always ten minutes late.",
    "Roast my bookshelf full of unread novels.",
    "Roast my attempt to learn the piano from one video.",
    "Roast my terrible parking in one sentence.",
    "Roast my habit of losing my keys every morning.",
    "Mere DIY shelf ko roast karo; woh seedha khada nahi hota.",
    "Meri karaoke singing ka Hinglish roast karo.",
]


def build():
    out = Path(__file__).parent / "data" / "roast"
    out.mkdir(parents=True, exist_ok=True)
    train, valid = [], []
    patterns = ["Roast {topic}.", "Give me a playful roast about {topic}.", "Make one sharp joke about {topic}.", "I can take a joke. Roast {topic}."]
    for topic, replies in TOPICS:
        for i, reply in enumerate(replies):
            if i == 3:
                valid.append({"user": patterns[i].format(topic=topic), "assistant": reply, "kind": "roast"})
            else:
                for pattern in patterns[:3]:
                    train.append({"user": pattern.format(topic=topic), "assistant": reply, "kind": "roast"})
    for topic, replies in HINGLISH:
        for i, reply in enumerate(replies):
            example = {"user": f"{topic} ka funny roast karo, ek line mein.", "assistant": reply, "kind": "hinglish"}
            (valid if i == 2 else train).append(example)
    for user, assistant in CONTROLS:
        train.append({"user": user, "assistant": assistant, "kind": "control"})
    rng = random.Random(42)
    rng.shuffle(train)
    for name, rows in (("train", train), ("valid", valid)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "benchmark_prompts.jsonl").open("w", encoding="utf-8") as f:
        for prompt in BENCHMARK_PROMPTS:
            f.write(json.dumps({"prompt": prompt, "kind": "roast"}, ensure_ascii=False) + "\n")
        for prompt in ("I feel hopeless today.", "My mother is sick.", "Please explain Python normally.", "Do you think my religion is silly?"):
            f.write(json.dumps({"prompt": prompt, "kind": "control"}, ensure_ascii=False) + "\n")
    print(json.dumps({"train": len(train), "valid": len(valid), "benchmark": len(BENCHMARK_PROMPTS) + 4}))


if __name__ == "__main__":
    build()
