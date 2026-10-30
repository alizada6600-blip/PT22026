# ReviewLens — English Product Review Sentiment Analyzer

Author: Mohammad Alizada
Student ID: 20243120081
Course: Professional Training — YN3012170034
University: Yunnan University, School of Software
Assigned folder: ali081

## Overview

ReviewLens predicts positive or negative sentiment in English product
reviews using TF-IDF and Logistic Regression. A Gradio interface lets
users enter reviews and view predictions and probability estimates.

## Installation

Run the following commands in a terminal inside the ali081 folder.
Use the same Python environment for installation, training, and the app.

Create an optional virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv/Scripts/activate

On macOS or Linux:

    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

Tested Python version: 3.13.9

## Recreate the model

Datasets and trained models are excluded from Git.

On a fresh checkout, run:

    python train_v2.py --output models/sentiment_pipeline_v2.joblib

The script downloads a sample of 20,000 review bodies from the official
Amazon Polarity training split, or reuses the local sample. It cleans
the data and trains the selected final model.

Internet access to Hugging Face is needed for the first download.
A GPU is not required.

The script refuses to overwrite an existing output model.
To reproduce separately when the application model already exists, run:

    python train_v2.py

The default output is:

    models/sentiment_pipeline_v2_reproduced.joblib

A JSON file records package versions, training counts, and the raw
sample's SHA-256 hash.

## Run the app

In the terminal, run:

    python app_v2.py

Open the local URL printed in the terminal.
Keep the terminal open while using the app. Press Ctrl+C to stop it.

The app loads models/sentiment_pipeline_v2.joblib.

## Features

- Positive and negative sentiment predictions.
- Model probability estimates.
- Example reviews and a Clear button.
- Helpful messages for empty or unrecognized input.
- A maximum input length of 5,000 characters.
- An uncertainty message when neither class reaches 60%.
- Colored result cards and a navy-and-teal interface.

The 60% threshold is a display rule, not a calibrated confidence threshold.

## Version 1 experiment

The original experiment used the Amazon portion of UCI's Sentiment
Labelled Sentences dataset.

After removing 13 duplicates, 987 sentences remained:
690 training, 148 validation, and 149 test sentences.

Naive Bayes was selected using validation Macro F1 and retrained on
838 training and validation sentences.

It achieved 81.88% accuracy on its original 149-sentence test set.

## Version 2 experiment

A buffered-shuffle sample of 20,000 review bodies was obtained from
fancyzhx/amazon_polarity using seed 42 and buffer size 10,000.
This is not a uniform random sample of the complete dataset.

Seven duplicates were removed, leaving 19,993 reviews:
15,994 training and 3,999 validation reviews.

TF-IDF was fitted on training data only during model comparison.

| Model | Validation accuracy | Validation Macro F1 |
|---|---:|---:|
| Naive Bayes | 86.15% | 0.8613 |
| Logistic Regression | 88.00% | 0.8799 |

Logistic Regression was selected and retrained on all 19,993
development reviews.

Final settings:
- TF-IDF ngram_range=(1, 2), min_df=2.
- max_features=100000, sublinear_tf=True.
- No stop-word removal.
- Logistic Regression C=10, solver=liblinear.
- max_iter=2000, random_state=42.

## Shared test evaluation

Both versions were evaluated on the same 5,000-review sample from the
official Amazon Polarity test split, using shuffle seed 123 and
buffer size 10,000.

Duplicate and conflicting texts were checked. Exact normalized matches
with previously used reviews were excluded; none were found in this
sample. Near-duplicates were not exhaustively checked.

| Model | Test accuracy | Test Macro F1 |
|---|---:|---:|
| Version 1 — Naive Bayes | 70.86% | 0.7055 |
| Version 2 — Logistic Regression | 87.56% | 0.8754 |

Version 2 correctly classified 4,378 of 5,000 reviews.

| Actual class | Predicted negative | Predicted positive |
|---|---:|---:|
| Negative | 2088 | 328 |
| Positive | 294 | 2290 |

The gain was 16.70 percentage points on this shared sample.
Both the dataset and classifier changed, so this measures the complete
system improvement rather than the effect of a single change.

## Reproducibility

The standalone script reproduced all 5,000 saved model predictions in
the original environment using the existing local training sample.

This confirms prediction agreement on that sample, not byte-identical
files or identical behavior on every possible input.

Fresh downloads may differ if upstream data or sampling software changes.

train_v2.py recreates the selected final model. The notebook records
the exploratory comparisons and evaluation. It also contains one-time
setup and Git commands; inspect these before execution rather than
blindly using Run All.

## Files

| File or folder | Purpose |
|---|---|
| app_v2.py | Current application |
| train_v2.py | Standalone model training |
| sentiment_analysis.ipynb | Development and evaluation |
| requirements.txt | Dependency versions |
| results/ | Evaluation tables and figures |
| data/ | Local datasets, excluded from Git |
| models/ | Local models, excluded from Git |

## Limitations

- Only positive and negative categories are supported.
- Neutral, mixed, sarcastic, and unrelated text may be misclassified.
- Unrelated input is not automatically detected.
- Probability estimates are not calibrated guarantees.
- Confident predictions can still be wrong.
- Amazon Polarity uses 1–2 stars as negative and 4–5 stars as positive.
  Three-star reviews are excluded.
- Review ratings may not represent every sentence's sentiment.
- Results describe the evaluated sample, not all product reviews.

## Sources

- UCI Sentiment Labelled Sentences:
  https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- Amazon Polarity dataset:
  https://huggingface.co/datasets/fancyzhx/amazon_polarity
- Zhang, Zhao, and LeCun (2015), Character-level Convolutional
  Networks for Text Classification:
  https://arxiv.org/abs/1509.01626

## Submission

Keep project changes within ali081.
Do not commit downloaded datasets or trained model files.
Submit through the required pull-request workflow.
