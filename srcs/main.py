from analyze import *
from utils import *
from pipeline import *


subjects = [1]
runs = {'do': [5, 9, 13],
		'imagine': [6, 10, 14]}


def	run_calls(options):
	if options.analyze:
		main_analyze(subjects, runs, options.visu)
	else:
		if not check_dumped_files_exists():
			raise Exception("You need to run the program with -a first in"\
								" order to dump the files needed for training"\
								" , prediction or pipeline creation.")
		if options.pipeline:
			create_pipeline(subjects, runs)
							

if __name__ == "__main__":
	options = parse_args()
	run_calls(options)
	# main_analyze(subjects, runs, options.visu)
	# create_pipeline(subjects, runs)
	
