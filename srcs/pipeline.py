import mne
import random
import matplotlib.pyplot as plt
import numpy as np
import time

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from CSP import CSP
from utils import load_saved_data
from analyze import main_analyze


def make_new_pipeline(model, model_name, csp, X, y):
	r_state = random.randint(0, 100)
	print("random_state {} for model {}".format(r_state, model_name))
	cv = ShuffleSplit(10, test_size=.2, random_state=r_state)
	clf = Pipeline([('CSP', csp), (model_name, model)])
	model_scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
	print('{} scores : {}'.format(model_name, model_scores))
	mean_model_scores, std_model_scores = np.mean(model_scores), np.std(model_scores)
	print("mean {} scores : {} | std : {}".format(model_name, mean_model_scores, std_model_scores), end='\n\n')
	#mean_scores_lda, std_scores_lda = np.mean(scores_lda), np.std(scores_lda)



def	create_pipeline(subjects, runs):
	#data, f_data, X, y = main_analyze([2, 3, 4, 5, 6, 7, 8, 9, 10], runs, False)
	data, filtered_data, X, y = load_saved_data()
	csp = CSP(n_components=3)
	svc = SVC(gamma='auto')
	#cv = ShuffleSplit(10, test_size=.2, random_state=random.randint(0, 101))

	params_grid = {'SVC__C': [10, 1, 100, 1000], 'SVC__gamma': [0.1, 1, 0.001, 0.0001],
					'SVC__kernel': ['linear', 'rbf']}
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
	#grid = GridSearchCV(svc, params_grid, refit=True, verbose=True)
	clf = Pipeline([('CSP', csp), ('SVC', svc)])
	grid_pipeline = GridSearchCV(clf, params_grid)
	grid_pipeline.fit(X_train, y_train)
	print(grid_pipeline.best_params_)
	y_hat = grid_pipeline.predict(X_test)
	print(classification_report(y_test, y_hat))

	#model_scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
	#print(f'LDA scores : {model_scores}')
	#print(f'mean acc : {np.mean(model_scores)} | std : {np.std(model_scores)}')
	#model_names = ['SVC', 'KNN', 'LDA']
	#for model, model_name in zip(models, model_names):
	#	make_new_pipeline(model_name, model, csp, X, y)

	"""cv = ShuffleSplit(10, test_size=.2, random_state=random.randint(0, 101))

	scores = []
	lda = LDA()
	lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
	svc = SVC(gamma='auto')
	xgb = XGBClassifier()
	knn = KNeighborsClassifier(weights='distance')

	csp = CSP(n_components=4)

	start_time = time.time()
	clf = Pipeline([('CSP', csp), ('LDA', lda)])
	scores_lda = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
	print('lda scores : ', scores_lda)
	mean_scores_lda, std_scores_lda = np.mean(scores_lda), np.std(scores_lda)

	start_time = time.time()
	#cv = ShuffleSplit(5, test_size=.2, random_state=42)
	clf2 = Pipeline([('CSP', csp), ('LDA', lda_shrinkage)])
	scores_lda_shrinkage = cross_val_score(clf2, X, y, cv=cv, n_jobs=1)
	print('scores lda_shrinkage : ', scores_lda_shrinkage)
	mean_scores_lda_shrinkage, std_scores_lda_shrinkage = np.mean(scores_lda_shrinkage), \
															np.std(scores_lda_shrinkage)
	print("done second pipeline in {:.2f}".format(time.time() - start_time))

	cv = ShuffleSplit(10, test_size=.2, random_state=random.randint(0, 101))
	start_time = time.time()
	clf = Pipeline([('CSP', csp), ('SVC', svc)])
	scores_svc = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
	print('scores svc : ', scores_svc)
	mean_scores_svc, std_scores_svc = np.mean(scores_svc), np.std(scores_svc)

	le = LabelEncoder()
	y_cpy = le.fit_transform(y)
	cv = ShuffleSplit(10, test_size=.2, random_state=random.randint(0, 101))
	start_time = time.time()
	clf = Pipeline([('CSP', csp), ('XGB', xgb)])
	scores_xgboost = cross_val_score(clf, X, y_cpy, cv=cv, n_jobs=1)
	print('scores xgboost : ', scores_xgboost)
	mean_scores_xgboost, std_scores_xgboost = np.mean(scores_xgboost), np.std(scores_xgboost)

	cv = ShuffleSplit(10, test_size=.2, random_state=random.randint(0, 101))
	clf = Pipeline([('CSP', csp), ('KNN', knn)])
	scores_knn = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
	print('scores knn : ', scores_knn)
	mean_scores_knn, std_scores_knn = np.mean(scores_knn), np.std(scores_knn)

	class_mean = np.mean(y == y[0])
	class_mean = max(class_mean, 1. - class_mean)

	print("LDA accuracy : {:.2f}".format(np.mean(scores_lda)))
	#print("LDA Shrinked accuracy : {:.2f}".format(np.mean(scores_lda_shrinkage)))
	print("SVC accuracy : {:.2f}".format(np.mean(scores_svc)))
	print("XGBOOST accuracy : {:.2f}".format(np.mean(scores_xgboost)))
	print("Knn accuracy : {:.2f}".format(np.mean(scores_knn)))"""
