import argparse
import mne


tmin, tmax = -1., 4.


from mne import events_from_annotations
from sklearn.model_selection import ShuffleSplit

def	parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--visu', default=False, action='store_true')
	options = parser.parse_args()
	return options

 
