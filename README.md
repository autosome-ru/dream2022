# DREAM 2022 Challenge| *autosome.org* reproduction repository

## Environment
To ensure that results of evaluating scripts explained below will deviate as little as possible from the results presented at the *DREAM 2022 Challenge*, we strongly advise to use our **conda** environment provided in the `environment.yml` file. The environment can be initialized via
```
> conda env create -f environment.yml
```
And then activated with 
```
> conda activate dream_autosome
```
Please note that we did the training on a GPU, and it is likely that the conda environment doesn't accomodate fully for a possible TPU usage.

## Input data

Before you proceed, make sure that `test_sequences.txt` file is in the root directiory of the repository. If you aim to re-train the model, make sure that the same applies for `train_sequences.txt` (this file is not provided with the repository as it is too large).

## Our submission

You can find predictions of our model that correspond to the following result from the leaderboard

|#|Competitor|Submission Time|Score PearsonR^2|Score Spearman|PearsonR^2|Spearman|
|-|----------|---------------|----------------|--------------|----------|--------|
|**1**|autosome.org|2022-07-31T19:39:12+03:00|0.829|0.860|0.952|0.979|


in the `submissions` folder. Namely, the files are `results.txt` and `results.json` -- they differ only in format and the latter is the file that was actually uploaded to the leaderboard system (here, we provide both variants for a convenience).

## Test-time evaluation
To reproduce predictions *almost* exactly (up to some floating point errors due to different GPUs) as they were present at *DREAM* leaderboard, just run the `test.sh` bash script. Then, predictions will be saved to the `results.txt` file in the root directory of the repository. If you want to obtain predictions in `json` format as was required by the leaderboard system, consider changing `--output_format tsv` and `--output results.txt` to `--output_format json` and `--output results.json` respectively.

## Training

You can re-train the model with `train.sh` script. Chances are, that your results will differ from those obtained at our local machine, however we have a hope at our hearts that those differences will not be drastic. Anyways, you might be interested in tweaking some of the arguments of the training to better evaluate the quality of our proposed model, namely
- `--seed` -- the seed for pseudo-random numbers generator (we use 42 as a default);
- `--model_dir` -- directory where the trained models will be stored (here, the script will save models to a `model_1`).

Note that models are saved each epoch, i.e. given the fact the we use 80 training epochs, your final model of interest will be the last one (`model_80.pth`).

You'll also probably would like to obtain predictions from newly re-trained models. For that purpose, you need to change those two arguments in the `test.sh` script:
- `--output` -- a path to a file where predictions will be stored;
- `--model` -- a path to a saved model `.pth`-file.

## Troubleshooting

So far we are aware only of a single issue that may or may not arise due to simultaneous openings of dataset files. The issue can be mitigated via setting `ulimit -n` to some high value, e.g.
```
> ulimit -n 1000000
```
