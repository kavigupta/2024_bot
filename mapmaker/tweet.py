import re
import pickle

import attr

import tweepy
import gspread

from .mapper import USAPresidencyBaseMap

from .SECRET import KEY

from .generate_image import get_althistory_image, get_image, get_gondor_image


@attr.s
class Tweeter:
    user_id = attr.ib()
    start_symbol = attr.ib()
    short_prefix = attr.ib()
    gdrive = attr.ib()
    get_image = attr.ib()

    def current_tweet_id(self, my_api):
        tweets = my_api.user_timeline(
            screen_name=self.user_id, count=10, include_rts=False, tweet_mode="extended"
        )
        for t in tweets:
            mat = re.match(rf"{self.start_symbol} (\d+)", t.full_text)
            if mat:
                return 1 + int(mat.group(1))
        return 1

    def post(self, name, pkl_path):
        with open(pkl_path, "rb") as f:
            state_margins = pickle.load(f)
        gc = gspread.service_account()
        sh = gc.open(self.gdrive)
        sh.sheet1.append_row([str(name)] + list(state_margins))

    def tweet_map(self):
        my_auth = tweepy.OAuthHandler(
            self.key["my_consumer_key"], self.key["my_consumer_secret"]
        )
        my_auth.set_access_token(
            self.key["my_access_token"], self.key["my_access_token_secret"]
        )
        my_api = tweepy.API(my_auth)
        number = self.current_tweet_id(my_api)
        image, pkl = self.get_image(number)
        self.post(f"{self.short_prefix}_{number}", pkl)
        message = f"{self.start_symbol} {number}"
        my_api.update_with_media(image, status=message)
        print("Tweeted message")

    @property
    def key(self):
        return KEY[self.user_id]


BOTS = [
    Tweeter(
        user_id="bot_2024",
        start_symbol="2024 scenario",
        short_prefix="scenario",
        gdrive="State Margins",
        get_image=lambda number: get_image(
            number, number, basemap=USAPresidencyBaseMap()
        ),
    ),
    Tweeter(
        user_id="bot_althistory",
        start_symbol="Alternate Universe",
        short_prefix="universe",
        gdrive="State Margins (Alternate History)",
        get_image=lambda number: get_althistory_image(number),
    ),
    Tweeter(
        user_id="ElectionsGondor",
        start_symbol="Gondor scenario",
        short_prefix="gondor_scenario",
        gdrive="State Margins (Gondor)",
        get_image=get_gondor_image,
    ),
]
