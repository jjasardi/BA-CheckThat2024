
import gc
import os
from pathlib import Path
import json
from dataclasses import asdict
from typing import List
import asyncio

import lmql
from lmql import LMQLResult

from checkthat2023.tasks.base import Sample
from checkthat2023.tasks.task1a import Task1ASample


@lmql.query
async def gpt_clf_simple(claim: str):
    '''
    argmax
       "Consider the following Tweet:\n"
       "\"{claim}\"\n\n"
       "Do you think this Tweet contains a claim that is worth fact-checking?\n\n"
       "Answer:[ANSWER]\n"
       "Reasoning:[REASON]\n"
    from
       "openai/text-davinci-003"
    where
       STOPS_AT(REASON, ".")
       and ANSWER in [ 'Yes', 'No']
    '''


class GPTClf:

    def __init__(
        self,
        secret_file: Path,
        cache_file: Path,
    ):
        self.cache_file = cache_file

        with secret_file.open('r') as fin:
            secrets = json.load(fin)
            os.environ['OPENAI_API_KEY'] = secrets['secret_key']
            del secrets

    async def get_single(
        self,
        sample: Task1ASample,
    ) -> LMQLResult:

        if self.cache_file.exists():
            with self.cache_file.open('r') as fin:
                cache = json.load(fin)
        else:
            cache = {}

        cached = cache.get(sample.id)
        if cached is not None:
            res = LMQLResult(**cached)
        else:
            res_list = await gpt_clf_simple(sample.tweet_text)
            res = res_list[0]
            cache[sample.id] = asdict(res)
            with self.cache_file.open('w') as fout:
                json.dump(fp=fout, obj=cache, indent=2)

        return res

    async def predict(
        self,
        samples: List[Task1ASample]
    ) -> List[Sample]:
        res = []

        for s in samples:
            r = await self.get_single(s)
            ns = Sample(
                id=s.id,
                class_label=r.variables['ANSWER'].strip().lower() == 'yes',
            )
            res.append(ns)
            del r
            gc.collect()

        return res
