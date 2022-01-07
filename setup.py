from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This MLOps project aims to use the Transformers framework from Hugging Face in order to tweak a pre-trained NLP model to accurately gauge the sentiment of an Amazon review (being able to guess the whether the rating of a product is positive or negative given only the text in a review).',
    author='group9 DTU MLops',
    license='MIT',
)
