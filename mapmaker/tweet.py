import tweepy

import numpy as np

from .SECRET import (
    my_consumer_key,
    my_consumer_secret,
    my_access_token,
    my_access_token_secret,
)

from .generate_image import sample_image


def tweet_map():
    word, image = sample_image()
    my_auth = tweepy.OAuthHandler(my_consumer_key, my_consumer_secret)
    my_auth.set_access_token(my_access_token, my_access_token_secret)
    my_api = tweepy.API(my_auth)
    message = f"2024 scenario: {word}"
    my_api.update_with_media(image, status=message)
    print("Tweeted message")
