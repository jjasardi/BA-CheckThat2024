
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


from checkthat2023.tasks.task1b import load
from checkthat2023.evaluation import evaluate, build_prediction_samples


def pipeline(seed: int = 0xdeadbeef) -> Pipeline:
    return Pipeline(
        steps=[
            ("tf-idf", TfidfVectorizer(
                ngram_range=(1, 5),
                min_df=3,
                binary=False,
                norm='l2',
                use_idf=True,
                sublinear_tf=True,
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
        sample.text
        for sample in dataset.train
    ]
    train_labels = [
        sample.class_label
        for sample in dataset.train
    ]

    clf = pipeline(0xdeadbeef)
    clf.fit(train_txts, train_labels)

    dev_texts = [
        sample.text
        for sample in dataset.dev_test
    ]
    pred = clf.predict(dev_texts)

    print(evaluate(gold=dataset.dev_test, prediction=build_prediction_samples(dataset.dev_test, pred)))


if __name__ == '__main__':
    main()
