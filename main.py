from analyze import *
from utils import *
from pipeline import *


subjects = [1]
runs = {'do': [5, 9, 13],
		'imagine': [6, 10, 14]}



if __name__ == "__main__":
	options = parse_args()
	data, f_data, train_data, labels = main_analyze(subjects, runs, options.visu)
	"""print(data)
	print(f_data)
	print(train_data)
	print(labels)"""
	
