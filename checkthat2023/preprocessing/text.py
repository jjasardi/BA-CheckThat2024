
import re
from typing import Callable

from emoji import demojize, replace_emoji
from nltk.tokenize import TweetTokenizer
from urlextract import URLExtract


class ReferenceTweetNormalizer(Callable[[str], str]):
    # official Tweet normalization provided by shared task organizers
    # from: https://gitlab.com/checkthat_lab/clef2023-checkthat-lab/-/blob/main/task1/baselines/TweetNormalizer.py

    def __init__(self):
        self.tokenizer = TweetTokenizer()

    @staticmethod
    def __normalize_token(token: str) -> str:
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return token

    def __call__(self, tweet_text: str) -> str:
        tokens = self.tokenizer.tokenize(tweet_text.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([ReferenceTweetNormalizer.__normalize_token(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

        return " ".join(normTweet.split())


class TweetNormalizer(Callable[[str], str]):

    def __init__(self):
        self.url_extractor = URLExtract()
        self.user_regex = re.compile(r'@\S+')

        self.url_token = "URL"
        self.user_token = "USER"
        self.emoji_token = "EMOJI"

    def __call__(self, tweet_text: str) -> str:
        res = tweet_text.lower()
        res = replace_emoji(res, self.emoji_token)
        res = self.user_regex.sub(self.user_token, res)
        urls = self.url_extractor.find_urls(res)
        for url in urls:
            res = res.replace(url, self.url_token)

        return res
