
from pathlib import Path

from checkthat2023.tasks.task1a import load
from checkthat2023.gptcheck import GPTClf

data_path = Path('./data')
secret_file = Path("/home/pius/work/dossma/prism_code/secrets/openai.json")
cache_file = Path("./cache/gpt-cache.json")

task1a = load(data_path, dev=False)

clf = GPTClf(secret_file=secret_file, cache_file=cache_file)

train_preds = await clf.predict(task1a.train)
dev_preds = await clf.predict(task1a.dev)
test_preds = await clf.predict(task1a.test)
