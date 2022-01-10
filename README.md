<p align="center">
  <a href="https://github.com/huggingface/transformers">
    <img src="https://camo.githubusercontent.com/b253a30b83a0724f3f74f3f58236fb49ced8d7b27cb15835c9978b54e444ab08/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f646f63756d656e746174696f6e2d696d616765732f7265736f6c76652f6d61696e2f7472616e73666f726d6572735f6c6f676f5f6e616d652e706e67" alt="Logo" width = "300" height = "60">
  </a>


</p>


## Table of contents

- [Project description & goal](#description-goal)
- [Framework](#framework)
- [Data](#data)
- [Models](#models)
- [Structure](#structure)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)


## Project description & goal

This MLOps project aims to use the Transformers framework from Hugging Face in order to tweak a pre-trained NLP model to accurately gauge the sentiment of an Amazon review (being able to guess the whether the rating of a product is positive or negative given only the text in a review).

If time allows, we will explore creating our own NLP model from scratch and attempting to compete with the optimal model found in Transformers.

## Framework

The [Transformers framework](https://github.com/huggingface/transformers) provides a wide array of different pre-trained NLP models that can be modified to a specific NLP task. 

## Data

The amazon review data of a few million text rows is sourced from [Kaggle](https://www.kaggle.com/bittlingmayer/amazonreviews).. The data was created with the intention of applying the Facebook fastText library, however we will be using it with Transformers. It is in text format and requires some degree of preprocessing.

## Models

Transformers pre-trained models have been proven to be very quick and effective for some cursory sentiment analysis. [This short post](https://towardsdatascience.com/sentiment-analysis-with-pretrained-transformers-using-pytorch-420bbc1a48cd) highlights how we can let Transformers do the logic from the back-end, however we also have the option to select a specific pre-trained model. 

A good starting point would be the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model, which is a good reference point accuracy-wise for our model that we can then improve on.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Creators

**the big man :necktie: **

- <https://github.com/samytessier>

**The "J" project (project to be announced at a later date)**

- <https://github.com/TheJProject>


**didier with a 'p'**  <img src="https://ih1.redbubble.net/image.805943027.3203/st,small,507x507-pad,600x600,f8f8f8.u1.jpg" alt="didier" height="35" width ="35">

- <https://github.com/paulseghers>

## TODO (non-obvious things we might forget)
- auto dowload the `.bz2` from kaggle with wget into `data/raw`
- create wrapper functions for code blocks that might be used twice
- add docstrings to wrapper functions for better documentability of what's going on
## Copyright and license


Code and documentation copyright 2011-2018 the authors. Code released under the [MIT License](https://reponame/blob/master/LICENSE).

## Future ref
<https://huggingface.co/docs/transformers/training>
<https://www.kaggle.com/muonneutrino/sentiment-analysis-with-amazon-reviews/notebook>
<https://en.wikipedia.org/wiki/The_Transformers_(TV_series)>
