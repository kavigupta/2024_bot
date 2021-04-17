import re
import pickle

import tweepy
import gspread

from .SECRET import (
    my_consumer_key,
    my_consumer_secret,
    my_access_token,
    my_access_token_secret,
)

from .generate_image import get_image

USER_ID = "bot_2024"


def current_tweet_id(my_api):
    tweets = my_api.user_timeline(
        screen_name=USER_ID, count=10, include_rts=False, tweet_mode="extended"
    )
    for t in tweets:
        mat = re.match(r"2024 scenario (\d+)", t.full_text)
        if mat:
            return 1 + int(mat.group(1))
    return 1


def post(name, pkl_path):
    with open(pkl_path, "rb") as f:
        state_margins = pickle.load(f)
    gc = gspread.service_account()
    sh = gc.open("State Margins")
    sh.sheet1.append_row([str(name)] + list(state_margins))


def tweet_map():
    my_auth = tweepy.OAuthHandler(my_consumer_key, my_consumer_secret)
    my_auth.set_access_token(my_access_token, my_access_token_secret)
    my_api = tweepy.API(my_auth)
    number = current_tweet_id(my_api)
    used, image, pkl = get_image(number, number)
    post(f"scenario_{number}", pkl)
    message = f"2024 scenario {number}"
    my_api.update_with_media(image, status=message)
    print("Tweeted message")
