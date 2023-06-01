import argparse
import numpy as np
import mne
from mne.io.edf import read_raw_edf
import os


def	load_saved_data():
	try:
		data = read_raw_edf('dumps/data.edf', preload=True, stim_channel='auto')
		filtered_data = read_raw_edf('dumps/filtered_data.edf', preload=True,
										stim_channel='auto')
		train_data = np.load('dumps/train_data.pkl', allow_pickle=True)
		labels = np.load('dumps/labels.pkl', allow_pickle=True)
		print('ok')
	except Exception:
		raise Exception("Error loading previously dumped files, check what has"\
							" been dumped.")
	return data, filtered_data, train_data, labels


def	check_dumped_files_exists():
	"""
	Just checking if analyze has already been run, therefore if we have
	correctly dumped the filed needed for training and pipeline creation
	"""
	if not os.path.isdir('dumps'):
		return False
	dumps = os.listdir('dumps')
	if 'data.edf' not in dumps or 'filtered_data.edf' not in dumps or \
		'train_data.pkl' not in dumps or 'labels.pkl' not in dumps:
		return False
	return True


def	parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--analyze', default=False, action='store_true')
	parser.add_argument('-v', '--visu', default=False, action='store_true')
	parser.add_argument('-t', '--train', default=False, action='store_true')
	parser.add_argument('-p', '--predict', default=False, action='store_true')
	parser.add_argument('-P', '--pipeline', default=False, action='store_true')
	options = parser.parse_args()
	return options

