# benchmark.py
# Run a comparative benchmark for each of the models at different 
# quantization.
# Python 3.9
# Windwos/MacOS/Linux


import argparse
import json
import os
import random
import shutil
from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import Dataset
import lancedb
from lancedb import DBConnection
import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from embed import load_data, load_model, clean_text


seed = 1234
random.seed(seed)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--split",
		choices=["all", "train", "test", "validation"],
		default="all",
		help="Which split to load for embedding. Default is 'all'.",
	)
	parser.add_argument(
		"--subset",
		type=int,
		default=-1,
		help="Whether to use a subset of the data and if so, how large. Default is -1 for the entire dataset.",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=1,
		help="How big should the batches be when embedding the text data. Default is 1.",
	)
	args = parser.parse_args()

	# Files.
	dataset_name = "google/wiki40b"
	dataset_lang = "en"
	folder = dataset_name.replace("/", "_")
	cache_dir = folder + "_tmp"
	splits = ["train", "test", "validation"]

	# Check for the dataset already being downloaded.
	if not os.path.exists(folder) or len(os.listdir(folder)) == 0:
		# Iterate through each splits.
		for split in splits:
			# Download the dataset to the cache folder.
			data = load_dataset(
				dataset_name, 
				dataset_lang,
				split=split,
				cache_dir=cache_dir,
			)

			# Save the dataset to a fixed folder.
			save_path = os.path.join(folder, split)
			os.makedirs(save_path, exist_ok=True)
			data.save_to_disk(
				save_path, 
			)

		# Delete the cache folder.
		shutil.rmtree(cache_dir)

	# Load config.
	with open("config.json", "r") as f:
		config = json.load(f)

	model_configs = config["models"]
	model_names = sorted(list(model_configs.keys()))

	# Load the dataset.
	data = load_data(folder, args.split)
	if args.subset > 0:
		data = data.select(range(args.subset))

	print(data)
	
	# Clean the text data.
	data = data.map(lambda x: {"cleaned_text": clean_text(x["text"])})

	# Detect GPU.
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"

	# Initialize vector DB.
	db = lancedb.connect(config["vector-search_config"]["db_uri"])
	
	# Iterate through each model and embed the vectors accordingly.
	for model_name in model_names:
		print(model_name)
		model_config = model_configs[model_name]

		# TODO:
		# Fix embedding call for hkunlp models. They are 
		# encoder-decoder models similar to T5, so calling the model 
		# itself is not sufficient. You'd have to do something like
		# model.encoder(**inputs).

		if "hkunlp" in model_name:
			continue

		# Load model and tokenizer.
		tokenizer, model = load_model(model_config, model_name, device)

		# Embed data to database.
		# embed_docs(
		# 	db, config, model_name, model_config, tokenizer, model, data, args.batch_size, device
		# )


	
	# Exit the program.
	exit(0)

if __name__ == '__main__':
	main()