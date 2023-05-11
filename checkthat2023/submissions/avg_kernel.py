
from pathlib import Path
from typing import List

from sklearn.svm import SVC

from checkthat2023.tasks.task1a import Task1A
from checkthat2023.kernel_stuff.kernel import KernelList, KernelData
from checkthat2023.submissions import create_submission, SubmissionEntry


def avg_kernel_submission(
    dataset: Task1A,
    kernel_folder: Path,
    run_name: str,
) -> List[SubmissionEntry]:
    kernel_list = KernelList(
        kernels=[
            KernelData.load_from(kf)
            for kf in kernel_folder.glob('*')
        ],
        name="avg-kernel"
    )

    y_train = [
        s.class_label
        for s in dataset.train
    ]

    svm = SVC(C=1., kernel='precomputed', class_weight='balanced')
    svm.fit(kernel_list.train(), y_train)

    test_preds = svm.predict(kernel_list.test())

    sub_data = [
        SubmissionEntry(
            tweet_id=s.id,
            class_label='Yes' if l == 1 else 'No',
            run_id=run_name,
        )
        for s, l in zip(dataset.test, test_preds)
    ]

    return sub_data


if __name__ == "__main__":
    from checkthat2023.tasks.task1a import load
    task1a = load(data_folder=Path('data'), dev=False)
    kernels = Path('kernel_data')
    run_name = "avg-kernel"
    sub_data = avg_kernel_submission(
        dataset=task1a,
        kernel_folder=kernels,
        run_name=run_name,
    )
    create_submission(
        sub_data=sub_data,
        sub_file=Path('submission') / "subtask1A_en.tsv"
    )
