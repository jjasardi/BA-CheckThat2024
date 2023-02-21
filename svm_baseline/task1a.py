
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


from datasets.task1a import load
from datasets.base import Sample
from preprocessing.text import TweetNormalizer
from evaluation import evaluate


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


def main():
    dataset = load()

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
        for sample in dataset.dev_test
    ]
    pred = clf.predict(dev_texts)

    preds = [
        Sample(
            id=s.id,
            class_label=lbl,
        )
        for s, lbl in zip(dataset.dev_test, pred)
    ]

    print(evaluate(gold=dataset.dev_test, prediction=preds))


if __name__ == '__main__':
    main()
