# Self-supervised methods in NLP

## Introduction

In this subproject we implement pretraining methods 
[MLM](https://arxiv.org/pdf/1810.04805.pdf), 
[GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) 
and [MASS](https://arxiv.org/pdf/1905.02450.pdf)
for [T5-small](https://arxiv.org/pdf/1910.10683.pdf) transformer and test them on downstream tasks such as QA, text
summarization, and machine translation.

We used 10% of [BookCorpus](https://huggingface.co/datasets/bookcorpus)
dataset for pretraining,[SQuAD](https://huggingface.co/datasets/squad) for QA, 
[XSum](https://huggingface.co/datasets/xsum) for summarization and 1% of
[WMT-16 [de-en]](https://huggingface.co/datasets/wmt16) for machine translation.

For measuring quality we used

* ExactMatch and F1-score metrics for QA;
* ROUGE-1 metrics for summarization;
* BLEU metrics for machine translation;

## Details

Experiments were conducted on local computer with GPU Nvidia RTX 3060 and 20Gb RAM.
Hyperparameters for pretraining were taken from the corresponding papers.
We used hyperparameters from [hugging face] tutorials for fine-tuning. We implemented
convenient model wrappers, that allow us use the same pretrained model for
different task. That was actually really important as we had 3 different
pretraining approaches and for each method we had 3 different downstream
tasks.

We reimplemented all steps described in papers and trained 19 models:
* As a baseline we trained 3 models for each downstream task.
* For MLM we pretrained 2 models: one for one epoch and one for 10 epochs.
For each pretrained model we fine-tuned one on downstream task (3*2=6 models in sum).
* For GPT we also pretrained two models, as model trained for one epoch showed
miserable quality. Then we fine-tuned each model for two downstream tasks,
as QA task needs encoder embeddings, but GPT actually has no encoder (2*2=4 models in sum).
* For MASS we pretrained one model and fine-tuned it on three tasks.
* We also employed MASS-like approach from original [T5] paper: authors offered
to mask several short intervals of tokens instead of one long interval, as
it was in MASS. We also implemented and fine-tuned this approach. 

## How to run

Firstly, you need to properly setup the corresponding config. Secondly,
to pretrain your model you should use for example the following command:

``python3 pretrain.py --pretext_task mlm --num_epochs 10 --batch_size 64``

Command arguments description:
* `pretext_task` one of {`mlm`, `gpt`, `mass`}.
* `num_epochs` surprisingly, number of train epoches (1 by default).
* `accum_steps` batch accumulation steps. How many batches the model processes
before performing optimization step (1 by default).
* `batch_size` batch size (16 by default).

The pretrained model will be stored by `./checkpoints/<PRETEXT_TASK>/`.

You also can tune your pretrained model supervisely on downstream task by
specifying the following command:

``python3 supervised.py --downstream_task summarization --pretext_model mlm 
--fine_tune --model_pretrain_path <PATH> --num_epochs 10 --batch_size 64``

As in pretraining case we has analogous hyperparameters `num_epochs`,
`batch_size` and `accum_steps`. But there are also extra parameters:
* `downstream_task` one of {`qa`, `summarization`, `translation`}.
* `pretext_model` the method used for model pretraining. Important if you
tune GPT. May be `gpt` or empty string (by default).
* `fine_tune` whether to fine tune pretrained model or use linear probing.
* `model_pretrain_path` path to your pretrained model.
* `model_warmup` number of steps before your model fully unfreeze. Useful if you
tune your model with large learning rate. In that case the gradients with respect to
untrained head can significantly corrupt body parameters of the model (0 by default).

The result model will be stored by `./checkpoints/<DOWNSTREAM TASK>/final_model/final_model.pth`.

## Results

| Model    | SQuAD (EM/F1)   | XSum (ROUGE) | WMT-16 [de-en] (BLEU) |
|----------|-----------------|--------------|-----------------------|
| Baseline | 0.099/0.097     | 18.31        | 0.26                  |
| MLM-1    | 0.12/0.12       | 18.1         | 0.227                 |
| MLM-10   | **0.158/0.159** | **20.84**    | 0.203                 |
| GPT-1    | -               | 18.59        | 0.227                 |
| GPT-10   | -               | 19.34        | 0.272                 |
| MASS     | 0.1/0.11        | 19.26        | **0.337**             |
| MASS T5  | 0.1/0.099       | 18.96        | 0.279                 |

We can observe that all methods were able to outperform the baseline on 
at least one of the metrics.

On QA task MLM showed the best performance as transformer encoder is the
most important part of model on this task. MLM once again demonstrated 
top-1 quality on the summarization task. This means, a well-trained 
encoder is more important than good decoder on summarization task.

Finally, MASS performed the best on machine translation. Top-2 and top-3
approaches are MASS-T5 and GPT, which means that properly trained decoder
is more important on this task.