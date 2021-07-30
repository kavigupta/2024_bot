from .tweet import BOTS

from sys import argv

user_id = argv[1]
print(f"Running bot for {user_id}")

for bot in BOTS:
    if bot.user_id == user_id:
        bot.tweet_map()
        break
else:
    raise RuntimeError("None of the bots matched")
