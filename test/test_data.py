import os	
import pytest

from test import _PATH_DATA
from datasets import load_from_disk #run with `pytest -W ignore::DeprecationWarning` 

folder = "data/processed"

# Define size of datasets
N_train = 3000
N_test = 1700

trainset = load_from_disk(_PATH_DATA + '/processed/train_processed_size_3000')
evalset = load_from_disk(_PATH_DATA + '/processed/eval_processed_size_1700')

@pytest.mark.skipif(not os.path.exists(_PATH_DATA+"/processed/"), reason="Data files not found")
def test_on_datasets_len():
	"""
	checks datasets have the right length
	"""
	assert len(trainset) == N_train, "length of dataset should be == {}".format(N_train)
	assert len(evalset) == N_test, "length of dataset should be == {}".format(N_train)

@pytest.mark.skipif(not os.path.exists(_PATH_DATA+"/processed/"), reason="Data files not found")
def test_well_constructed():
	"""
	check if the training dataset has everything we need
	"""
	needed_features = {'content', 'input_ids', 'label', 'title'}
	assert needed_features.issubset(set(trainset.features.keys())), "all necessary features should be present in dataset"
@pytest.mark.skipif(not os.path.exists(_PATH_DATA+"/processed/"), reason="Data files not found")
def test_all_labels_present():
	for i in range(2): #labels are digits 1 or 0
		t = 2 #
		idx = 0
		while t != i:
			t = trainset[idx]['label']
			idx += 1
		assert idx < N_train, "all labels should appear at least once in train set"

