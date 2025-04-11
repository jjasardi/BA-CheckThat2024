
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


from checkthat2023.tasks.task1a import load
from checkthat2023.preprocessing.text import TweetNormalizer
from checkthat2023.evaluation import evaluate, build_prediction_samples


def pipeline(seed: int = 0xdeadbeef) -> Pipeline:
    return Pipeline(
        steps=[
            ("tf-idf", TfidfVectorizer(
                preprocessor=TweetNormalizer(),
                ngram_range=(1, 2),
                min_df=3,
                binary=True,
                norm='l2',
                use_idf=True,
                smooth_idf=True,
            )),
            ("svm", LinearSVC(
                C=1.0,
                class_weight='balanced',
                random_state=seed,
            )),
        ],
    )


def main(data_path: Path):
    dataset = load(data_path)

    train_txts = [
        sample.tweet_text
        for sample in dataset.train
    ]
    train_labels = [
        sample.class_label
        for sample in dataset.train
    ]

    clf = pipeline(0xdeadbeef)
    clf.fit(train_txts, train_labels)

    dev_texts = [
        sample.tweet_text
        for sample in dataset.dev
    ]
    pred = clf.predict(dev_texts)

    print(evaluate(gold=dataset.dev, prediction=build_prediction_samples(dataset.dev, pred)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", dest="data", type=Path, required=True)
    args = parser.parse_args()

    main(data_path=args.data)
