import mne
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from CSP import CSP
from utils import load_saved_data


def	create_pipeline(subjects, runs):
	# data, f_data, X, y = main_analyze(subjects, runs, False)
	data, f_data, X, y = load_saved_data()
	print(data)
	print(f_data)
	print(X)
	print(y)
	
