import os
import h5py
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

DATA_DIR = "raw_data"
FINAL_DATA = "data"

dataframes = []
folders = os.listdir(DATA_DIR)
for folder in folders:
	tsv_path = DATA_DIR + "/" + folder + "/airsim_rec.txt"
	tsv = pd.read_csv(tsv_path, sep = "\t")
	tsv["Folder"] = folder
	last_column = tsv['ImageFile'].str.split(";").tolist()
	if len(last_column[0]) > 1:
		image_name = [x[0] for x in last_column]
		tsv['ImageFile'] = image_name
	dataframes.append(tsv)
	
dataset = pd.concat(dataframes, axis = 0)
print("Number of data points: {0}".format(dataset.shape[0]))

train_val_test_split = [0.7, 0.2, 0.1]

mappings = {}
for i in range(1, dataset.shape[0] - 1):
	if dataset.iloc[i - 1]['Brake'] != 0:
		continue
	normalized_steering = [float(dataset.iloc[i - 1][['Steering']]) + 1 / 2.0]
	normalized_throttle = [float(dataset.iloc[i - 1][['Throttle']])]
	normalized_speed = [float(dataset.iloc[i - 1][['Speed']]) / 30]
	previous_state = normalized_steering + normalized_throttle + normalized_speed
	
	current_normalized_steering = [float(dataset.iloc[i][['Steering']]) + 1 / 2.0]
	next_normalized_steering = [float(dataset.iloc[i + 1][['Steering']]) + 1 / 2.0]
	
	current_label = [(normalized_steering[0] + current_normalized_steering[0] + next_normalized_steering[0]) / 3.0]
	
	image_path = DATA_DIR + "/" + dataset.iloc[i]['Folder'] + "/images/" + dataset.iloc[i]['ImageFile']
	if image_path not in mappings:
		mappings[image_path] = (current_label, previous_state)
		
datamap = [(key, mappings[key]) for key in mappings]
random.shuffle(datamap)

train_split = int(len(datamap) * train_val_test_split[0])
val_split = train_split + int(len(datamap) * train_val_test_split[1])

train_data = datamap[:train_split]
val_data = datamap[train_split : val_split]
test_data = datamap[val_split:]

print("Length of training data = {0}".format(len(train_data)))
print("Length of validation data = {0}".format(len(val_data)))
print("Length of testing data = {0}".format(len(test_data)))

print("Storing dataset in h5py")

data = [train_data, val_data, test_data]
file_names = ["train.h5", "val.h5", "test.h5"]

for d, fi in zip(data, file_names):
	path = FINAL_DATA + "/" + fi
	print(path)
	row = 0
	with h5py.File(path, 'w') as f:
		dataset_images = f.create_dataset('image', shape = (32, 144, 256, 3), maxshape = (None, 144, 256, 3), chunks = (32, 144, 256, 3), dtype = np.uint8)
		dataset_labels = f.create_dataset('label', shape = (32, 1), maxshape = (None, 1), chunks = (32, 1), dtype = np.float64)
		dataset_previous_state = f.create_dataset('previous_state', shape = (32, 3), maxshape = (None, 3), chunks = (32, 3), dtype = np.float64)
		for id in range(0, len(d), 32):
			chunk = d[id : id + 32]
			if len(chunk) == 32:
				image_path_chunk = [a for (a, b) in chunk]
				labels_chunk = np.asarray([b[0] for (a, b) in chunk])
				labels_chunk = labels_chunk.astype(float)
				previous_state_chunk = np.asarray([b[1] for (a, b) in chunk])
				previous_state_chunk = previous_state_chunk.astype(float)
				image_chunk = []
				for image_path in image_path_chunk:
					image = np.asarray(Image.open(image_path))
					image = image[:, :, :3]
					image_chunk.append(image)
				image_chunk = np.asarray(image_chunk)
				dataset_images[row:] = image_chunk
				dataset_labels[row:] = labels_chunk
				dataset_previous_state[row:] = previous_state_chunk
				row += image_chunk.shape[0]
			if id + 32*2 <= len(d):
				dataset_images.resize(row + image_chunk.shape[0], axis = 0)
				dataset_labels.resize(row + labels_chunk.shape[0], axis = 0)
				dataset_previous_state.resize(row + previous_state_chunk.shape[0], axis = 0)