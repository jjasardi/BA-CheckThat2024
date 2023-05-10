
import asyncio
from pathlib import Path

from checkthat2023.tasks.task1a import load
from checkthat2023.gptcheck import GPTClf

data_path = Path('./data')
secret_file = Path("/home/pius/work/dossma/prism_code/secrets/openai.json")
cache_file = Path("./cache/gpt-cache.json")

task1a = load(data_path, dev=False)

clf = GPTClf(secret_file=secret_file, cache_file=cache_file)


async def dew_it():
    _ = await clf.predict(task1a.train)
    _ = await clf.predict(task1a.dev)
    _ = await clf.predict(task1a.test)

loop = asyncio.get_event_loop()
loop.run_until_complete(dew_it())
