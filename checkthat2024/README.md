# Task 1: Check-worthiness Estimation in Text

The aim of this task is to evaluate whether a statement, sourced from either a tweet or a political debate, warrants fact-checking. To make this decision, one must consider questions such as ''Does it contain a verifiable factual claim?'' and ''Could it be harmful?'' before assigning a final label for its check-worthiness.

This year, we are offering multi-genre data: the tweets and/or transcriptions should be judged based solely on the text. The task is available in Arabic, English, and Spanish.


__Table of contents:__

- [Submission Guidelines](#submission-guidelines)
- [List of Versions](#list-of-versions)
- [Contents of the Directory](#contents-of-the-directory)
- [File Format](#file-format)
	- [Check-Worthiness of multigenre unimodal content](#subtask-1b-multigenre-unimodal-text-check-worthiness-of-multigenre-unimodal-content)
		- [Input Data Format - Tweets](#input-data-format-unimodal-text-tweets)
		- [Input Data Format - Political Debates](#input-data-format-unimodal-text-political-debates)
	- [Output Data Format](#output-data-format)
- [Format Checkers](#format-checkers)
- [Scorers](#scorers)
- [Baselines](#baselines)
- [Credits](#credits)



<!-- ## Evaluation Results
Submitted results will be available after the system submission deadline.
Kindly find the leaderboard released in this google sheet, [link](http://shorturl.at/nuCOS). you can find in the tab labeled "Task 1".
 -->

<!-- ## Submission Guidelines:
- Make sure that you create one account for each team, and submit runs through one account only.
- The last file submitted to the leaderboard will be considered as the final submission.
- Name of the output file has to be `subtask1[A/B]_lang.tsv` with `.tsv` extension (e.g., subtask1B_arabic.tsv); otherwise, you will get an error on the leaderboard. Subtasks are 1A, 1B. For subtask 1A, there are two languages (Arabic, and English). For subtask 1B, there are three languages (Arabic, Spanish, and English).
- You have to zip the tsv, `zip subtask1B_arabic.zip subtask1B_arabic.tsv` and submit it through the codalab page.
It is required to submit the team name and method description for each submission. **Your team name here must EXACTLY match that used during CLEF registration.**
- You are allowed to submit max 200 submissions per day for each subtask.
- We will keep the leaderboard private till the end of the submission period, hence, results will not be available upon submission. All results will be available after the evaluation period.

**Please submit your results on test data here: https://codalab.lisn.upsaclay.fr/competitions/12936** -->


## List of Versions

* __[2022/11/23]__
  - Training/Dev/Dev_Test data released for Arabic, English and Spanish.


## Contents of the Directory
* Main folder: [data](./data)
  	This directory contains files for all languages and subtasks.

* Main folder: [baselines](./baselines)<br/>
	Contains scripts provided for baseline models of the tasks.
* Main folder: [format_checker](./format_checker)<br/>
	Contains scripts provided to check format of the submission file.
* Main folder: [scorer](./scorer)<br/>
	Contains scripts provided to score output of the model when provided with label (i.e., dev and dev_test sets).

* [README.md](./README.md) <br/>
	This file!


## File Format


### Check-Worthiness of multigenre content


#### Input Data Format (Unimodal - Text -- Tweets)
For **Arabic**, and **Spanish** we use the same data format in the train, dev and dev_test files. Each file is TAB seperated (TSV file) containing the tweets and their labels. The text encoding is UTF-8. Each row in the file has the following format:

> tweet_id <TAB> tweet_url <TAB> tweet_text <TAB> class_label

Where: <br>
* tweet_id: Tweet ID for a given tweet given by Twitter <br/>
* tweet_url: URL to the given tweet <br/>
* tweet_text: content of the tweet <br/>
* class_label: *Yes* and *No*


**Examples:**
> 1235648554338791427	https://twitter.com/A6Asap/status/1235648554338791427	COVID-19 health advice⚠️ https://t.co/XsSAo52Smu	No<br/>
> 1235287380292235264	https://twitter.com/ItsCeliaAu/status/1235287380292235264	There's not a single confirmed case of an Asian infected in NYC. Stop discriminating cause the virus definitely doesn't. #racist #coronavirus https://t.co/Wt1NPOuQdy	Yes<br/>
> 1236020820947931136	https://twitter.com/ddale8/status/1236020820947931136	Epidemiologist Marc Lipsitch, director of Harvard's Center for Communicable Disease Dynamics: “In the US it is the opposite of contained.' https://t.co/IPAPagz4Vs	Yes<br/>
> ... <br/>

Note that the gold labels for the task are the ones in the *class_label* column.



#### Input Data Format (Unimodal - Text -- Political debates)
For **English** we use the same data format in the train, dev and dev_test files. Each file is TAB seperated (TSV file) containing the tweets and their labels. The text encoding is UTF-8. Each row in the file has the following format:

> sentence_id <TAB> text <TAB> class_label

Where: <br>
* sentence_id: sentence id for a given political debate <br/>
* text: sentence's text <br/>
* class_label: *Yes* and *No*


**Examples:**
> 30313	And so I know that this campaign has caused some questioning and worries on the part of many leaders across the globe.	No<br/>
> 19099	"Now, let's balance the budget and protect Medicare, Medicaid, education and the environment."	No<br/>
> 33964	I'd like to mention one thing.	No<br/>
> ... <br/>

Note that the gold labels for the task are the ones in the *class_label* column.


### Output Data Format
For both subtasks **1A**, and **1B** and for all languages (**Arabic**, **English**, and **Spanish**) the submission files format is the same.

The expected results file is a list of tweets/transcriptions with the predicted class label.

The file header should strictly be as follows:

> **id <TAB> class_label <TAB> run_id**

Each row contains three TAB separated fields:

> tweet_id or id <TAB> class_label <TAB> run_id

Where: <br>
* tweet_id or id: Tweet ID or sentence id for a given tweet given by Twitter or coming from political debates given in the test dataset file. <br/>
* class_label: Predicted class label for the tweet. <br/>
* run_id: String identifier used by participants. <br/>

Example:
> 1235648554338791427	No  Model_1<br/>
> 1235287380292235264	Yes  Model_1<br/>
> 1236020820947931136	No  Model_1<br/>
> 30313	No  Model_1<br/>
> ... <br/>


## Format Checkers

The checker for the task is located in the [format_checker](./format_checker) module of the project.
To launch the checker script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

The format checker verifies that your generated results files complies with the expected format.
To launch it run:

> python3 format_checker/subtask_1.py --pred-files-path <path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

`--pred-files-path` is to be followed by a single string that contains a space separated list of one or more file paths.

__<path_to_result_file_n>__ is the path to the corresponding file with participants' predictions, which must follow the format, described in the [Output Data Format](#output-data-format) section.

Note that the checker can not verify whether the prediction files you submit contain all tweets, because it does not have access to the corresponding gold file.


## Scorers

The scorer for the task is located in the [scorer](./scorer) module of the project.
To launch the script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

Launch the scorer for the subtask as follows:
> python3 scorer/subtask_1.py --gold-file-path=<path_gold_file> --pred-file-path=<predictions_file> --subtask=<name_of_the_subtask> --lang=<language><br/>

`--subtask` expects one of two options **A** or **B** to indicate the subtask for which to score the predictions file.

`--lang` expects one of three options **arabic** or **english** or **spanish** to indicate the language for which to score the predictions file.

The scorer invokes the format checker for the task to verify the output is properly shaped.
It also handles checking if the provided predictions file contains all tweets from the gold one.


## Baselines

The [baselines](./baselines) module currently contains a majority, random and a simple n-gram baseline.

<!-- **Baseline Results for Task 1 subtasks on Dev_Test**
|Model|subtask-1A--Arabic|subtask-1A--English|subtask-1B--Arabic|subtask-1B--Spanish|subtask-1B--English|
|:----|:----|:----|:----|:----|:----|
|Random Baseline |0.405|0.283|0.429|0.230|0.306|
|Majority Baseline|0.000|0.000|0.000|0.000|0.000|
|n-gram Baseline|0.491|0.645|0.202|0.511|0.821|
|Multimodal:<br/>ResNet+BERT SVM|0.416|0.442|NA|NA|NA| -->


To launch the baseline script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

To launch the baseline script run the following:
> python3 baselines/subtask_1b.py --train-file-path=<path_to_your_training_data> --dev-file-path=<path_to_your_test_data_to_be_evaluated> --lang=<language_of_the_subtask><br/>
```
python3 baselines/subtask_1b.py --train-file-path=data/CT23_1B_checkworthy_arabic/CT23_1B_checkworthy_arabic_train.tsv --dev-file-path=data/CT23_1B_checkworthy_arabic/CT23_1B_checkworthy_arabic_dev.tsv -l arabic
```


All baselines will be trained on the training dataset and the performance of the model is evaluated on the dev_test.

<!-- ## Submission guidelines
Please follow the submission guidelines discussed here: https://sites.google.com/view/clef2022-checkthat/tasks/task-1-identifying-relevant-claims-in-tweets?#h.sn4sm5zguq98.
 -->
## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2024/task1/

<!-- Contact:   clef-factcheck@googlegroups.com -->
