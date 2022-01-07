<p align="center">
  <a href="https://github.com/huggingface/transformers">
    <img src="https://camo.githubusercontent.com/b253a30b83a0724f3f74f3f58236fb49ced8d7b27cb15835c9978b54e444ab08/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f646f63756d656e746174696f6e2d696d616765732f7265736f6c76652f6d61696e2f7472616e73666f726d6572735f6c6f676f5f6e616d652e706e67" alt="Logo">
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

The amazon review data is sourced from [Kaggle](https://www.kaggle.com/bittlingmayer/amazonreviews). It consists of a few million text rows of Amazon customer reviews. The data was created with the intention of applying the Facebook fastText library, however we will be using it with Transformers. It is in text format and requires some degree of preprocessing.

## Models

Transformers pre-trained models have been proven to be very quick and effective for some cursory [sentiment analysis](https://towardsdatascience.com/sentiment-analysis-with-pretrained-transformers-using-pytorch-420bbc1a48cd). This short post highlights how we can let Transformers do the logic from the back-end, however we also have the option to select a specific pre-trained model. 

A good starting point would be the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model, which is a good reference point accuracy-wise for our model that we can then improve on.

## Structure

```working directory tree
group_9_mlops/
├── data
│   ├── test.ft.txt
│   └── train.ft.txt
└── README.md
```


## Creators

**Samy Tessier**

- <https://github.com/samytessier>

**The "J" project (project to be announced at a later date)**

- <https://github.com/TheJProject>


**didier**  <img src="https://ih1.redbubble.net/image.805943027.3203/st,small,507x507-pad,600x600,f8f8f8.u1.jpg" alt="didier" height="35" width ="35">


- <https://github.com/paulseghers>


## Copyright and license


Code and documentation copyright 2011-2018 the authors. Code released under the [MIT License](https://reponame/blob/master/LICENSE).

## Future ref
<https://huggingface.co/docs/transformers/training>
<https://www.kaggle.com/muonneutrino/sentiment-analysis-with-amazon-reviews/notebook>