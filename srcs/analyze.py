import mne
import matplotlib.pyplot as plt
import os
import warnings

from mne import pick_types, events_from_annotations, annotations_from_events
from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne.io.edf import read_raw_edf
from mne.channels import make_standard_montage

warnings.filterwarnings("ignore")


# EVENTS = 1: rest, 2: do-feet, 3: do-hands, 4: imagine-feet, 5: imagine-hands

EVENT_IDS = dict(T0=1, T1=2, T2=3)
TMIN, TMAX = -1., 4.


def	main_analyze(subjects, runs, visu):
	data = load_data(subjects, runs)
	if visu:
		raw_files_info(data)
	montage = set_montage(data)
	f_data = filter_data(data, montage)
	if visu:
		plot_data(data, f_data)
		channels_frequency_plot(data, f_data)
	train_data, labels = get_events(f_data)
	if not os.path.isdir('dumps'):
		os.makedirs('dumps')
	#data.save('dumps/data.fif', overwrite=True)
	#f_data.save('dumps/filtered_data.fif', overwrite=True)
	mne.export.export_raw('dumps/data.edf', data, overwrite=True)
	mne.export.export_raw('dumps/filtered_data.edf', f_data, overwrite=True)
	train_data.dump('dumps/train_data.pkl')
	labels.dump('dumps/labels.pkl')
	return data, f_data, train_data, labels


def	get_events(f_data):
	"""
	Loading and formatting events
	"""
	# Extract events informations from filtered data
	events, events_id = events_from_annotations(f_data)
	# Select only relavent channels from filtered data, in this case EEG channel
	picks = mne.pick_types(f_data.info, meg=False, eeg=True, stim=False, eog=False,
							exclude='bads')
	# Creating epochs for filtered data based on events information and their
	# 	corresponding ids
	# 	proj=True -> apply spatial filter or projector to EEG data, those are
	#				 used to remove or minimize unwanted artifacts or noise.
	# These epochs are time window centered around specific event or stimuli
	epochs = mne.Epochs(f_data, events, events_id, TMIN, TMAX, proj=True, picks=picks,
							baseline=None, preload=True)
	# Extracting labels associated with each event
	labels = epochs.events[:, -1]
	# Retrieving epoch data and returns the result as a numpy array
	epochs_data_train = epochs.get_data()
	return epochs_data_train, labels 
	
	
def	plot_data(data, f_data):
	plot = mne.viz.plot_raw(data, scalings={'eeg': 75e-6})
	plt.title("Unfiltered data")
	plot2 = mne.viz.plot_raw(f_data, scalings={'eeg': 75e-6})
	plt.title("Filtered data")
	plt.show()


def	raw_files_info(files):
	print(files)
	print(files.info)
	print(files.info['ch_names'])
	print(files.annotations)


def	load_data(subjects, runs):
	files = []
	for subject in subjects:
		print(f'\033[33mloading subject {subject}\033[0m')
		for i, j in zip(runs['do'], runs['imagine']):
			# Loading do and imagine runs for each subject
			files_d = [read_raw_edf(f, preload=True, stim_channel='auto') for f \
						in eegbci.load_data(subject, i)]
			files_i = [read_raw_edf(f, preload=True, stim_channel='auto') for f \
						in eegbci.load_data(subject, j)]
			
			# Fusing runs
			raw_d = concatenate_raws(files_d)
			raw_i = concatenate_raws(files_i)

			# Assocating execution runs with corresponding events
			events, _ = events_from_annotations(raw_d, event_id=dict(T0=1, T1=2, T2=3))
			mapping = {1: 'rest', 2: 'do-feet', 3: 'do-hands'}
			events_anno = annotations_from_events(events=events, event_desc=mapping, \
								sfreq=raw_d.info['sfreq'], orig_time=raw_d.info['meas_date'])
			raw_d.set_annotations(events_anno)
			raw_d.rename_channels(lambda x: x.strip('.'))

			# Associating imagine runs with corresponding events
			events, _ = events_from_annotations(raw_i, event_id=dict(T0=1, T1=2, T2=3))
			mapping = {1: 'rest', 2: 'imagine-feet', 3: 'imagine-hands'}
			events_anno = annotations_from_events(events=events, event_desc=mapping, \
							sfreq=raw_i.info['sfreq'], orig_time=raw_i.info['meas_date'])

			raw_i.set_annotations(events_anno)
			raw_i.rename_channels(lambda x: x.strip('.'))
			files.append(raw_d)
			files.append(raw_i)
	files = concatenate_raws(files)
	files.rename_channels(lambda x: x.strip('.'))
	return files


def	set_montage(files, plot=False):
	montage = make_standard_montage('standard_1020')
	eegbci.standardize(files)
	files.set_montage(montage)

	montage = files.get_montage()
	if plot:
		plot = montage.plot()
		plot = mne.viz.plot_raw(files, scalings={'eeg': 75e-6})
		plt.title("Unfiltered data")
		plt.show()
	return montage


def	filter_data(files, montage):
	cpy = files.copy()
	cpy.set_montage(montage)
	cpy.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
	return cpy


def	channels_frequency_plot(data, f_data):
	data.compute_psd().plot(average=False)
	plt.title("Unfiltered data")
	data.compute_psd().plot(average=True)
	plt.title("Averaged channels frequency - Unfiltered data")
	f_data.compute_psd().plot(average=False)
	plt.title("Filtered data")
	f_data.compute_psd().plot(average=True)
	plt.title("Averaged channels frequency - Filtered data")
	plt.show()


