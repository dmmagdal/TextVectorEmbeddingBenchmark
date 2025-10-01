# embed.py
# Embed the dataset to vectors at different quantization levels for all
# the models in the config file.
# Python 3.9
# Windwos/MacOS/Linux


import copy
from datetime import timedelta
import json
import math
import os
import shutil
from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import Dataset
import lancedb
from lancedb import DBConnection
import numpy as np
import pyarrow as pa
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def vector_preprocessing(
		article_text: str, 
		config: Dict, 
		model_config: Dict[str, Union[str, int]],
		tokenizer: AutoTokenizer, 
		recursive_split: bool = False
) -> List[Dict]:
	'''
	Preprocess the text to yield a list of chunks of the tokenized 
		text. Each chunk is the longest possible set of text that can 
		be passed to the embedding model tokenizer.
	@param: text (str), the raw text that is to be processed for
		storing to vector database.
	@param: config (dict), the configuration parameters. These 
		parameters detail important parts of the vector preprocessing
		such as context length.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: recursive_split (bool), whether to use the recursive 
		splitting scheme for long text chunks or a more basic one. Is
		false by default.
	@return: returns a List[Dict] of the text metadata. This metadata 
		includes the split text's token sequence, index (with respect
		to the input text), and length of the text split for each split
		in the text.
	'''
	# Pull the model's context length and overlap token count from the
	# configuration file.
	# model_name = config["vector-search_config"]["model"]
	# model_config = config["models"][model_name]
	context_length = model_config["max_tokens"]
	overlap = config["preprocessing"]["token_overlap"]

	# Make sure that the overlap does not exceed the model context
	# length.
	# assert overlap < context_length, f"Number of overlapping tokens ({overlap}) must NOT exceed the model context length ({context_length})"

	# NOTE:
	# Initially there were plans to have text tokenized and chunked by
	# token (chunk lengths would be context_length with overlap number
	# of tokens overlapping). This proved to be more complicated than
	# thought because it required tokens be decoded back to the
	# original text exactly, something that is left up to the
	# implementation of each model's tokenizer. To allow for support of
	# so many models, there had to be a more general method to handle
	# text tokenization while keeping track of the original text
	# metadata. 

	# NOTE:
	# Splitting scheme 1 (recursive split):
	# 1) Split into paragraphs (split by newline ("\n\n", "\n") 
	#	characters). This is covered by the high_level_split() 
	#	recursive function.
	# 2) Split paragraphs that are too long (split by " " (word level 
	#	split) and "" (character level split)). This is covered by the
	#	low_level_split() recursive function that is called by the
	#	high_level_split() recursive function when such is the case.
	# Splitting scheme 2 (direct/basic split):
	# 1) Split into paragraphs (split by newline ("\n\n", "\n") 
	#	characters). 
	# 2) Split paragraphs that are too long (split by token lengths
	#	with some overlap, recycle the text metadata for all chunks in
	# 	that paragraph).

	# Initialize splitters list and text metadata list. The splitters
	# are the same as default on RecursiveCharacterTextSplitter.
	splitters = ["\n\n", "\n", " ", ""] 
	metadata = []

	# Add to the metadata list by passing the text to the high level
	# recursive splitter function.
	if recursive_split:
		metadata += high_level_split(
			article_text, 0, tokenizer, context_length, splitters
		)
	else:
		metadata = direct_split(
			article_text, overlap, tokenizer, context_length, splitters[0]
		)
	
	# Return the text metadata.
	return metadata


def direct_split(
		text: str, 
		offset: int, 
		tokenizer: AutoTokenizer, 
		context_length: int, 
		splitter: str
) -> List[Dict]:
	assert offset < context_length, \
		f"offset ({offset}) must be less than the maximum context length ({context_length})"
	
	# Initialize the metadata list.
	metadata = []

	# Split the text.
	text_splits = text.split(splitter)

	# Iterate through the list 
	for split in text_splits:
		# Skip the split if it is an empty string.
		if split == "":
			continue

		# Get the split metadata (index with respect to original text 
		# plus offset and split length).
		split_idx = text.index(split) #+ offset
		split_len = len(split)

		# Tokenize the split.
		tokens = tokenizer.encode(split, add_special_tokens=False)

		if len(tokens) <= context_length:
			# If the token sequence is less than or equal to the 
			# context length, tokenize the text split again (this time
			# with padding), and add the entry to the metadata.
			tokens = tokenizer.encode(
				split, 
				add_special_tokens=False, 
				padding="max_length"
			)
			metadata.append({
				"tokens": tokens,
				"text_idx": split_idx,
				"text_len": split_len
			})
		else:
			# If the token sequence is greater than the context length,
			# split the embeddings/tokens with some overlap and recycle
			# the text metadata for all splits.
			step = context_length - offset
			pad_token_id = tokenizer.pad_token_id

			for start in range(0, len(tokens), step):
				end = start + context_length
				chunk = tokens[start:end]

				# Pad last chunk if shorter than context_length
				if len(chunk) < context_length:
					chunk = chunk + [pad_token_id] * (context_length - len(chunk))

				metadata.append({
					"tokens": chunk,
					"text_idx": split_idx,
					"text_len": split_len
				})

				if end >= len(tokens):
					break

	assert(
		all(len(data["tokens"]) == context_length for data in metadata)
	), f"Expected all tokens to be the {context_length} long."

	# Return the metadata.
	return metadata


def high_level_split(
		text: str, 
		offset: int, 
		tokenizer: AutoTokenizer, 
		context_length: int, 
		splitters: List[str]
) -> List[Dict]:
	'''
	(Recursively) split the text into paragraphs and extract the 
		metadata from the text slices of the input text. If the 
		paragraphs are too large, call the low_level_split() recursive 
		function and extract the metadata from there too.
	@param: text (str), the text that is to be processed for storing to
		vector database.
	@param: offset (int), the index of the input text with respect to
		the original text.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: context_length (int), the maximum number of tokens 
		supported by the model. This helps us chunk the text if the 
		tokenized output is "too long".
	@param: splitters (List[str]), the list of strings that will be 
		used to split the text. For this function, we expect the 
		"top-most" strings to be either in the set ("\n\n", "\n").
	@return: returns a List[Dict] of the text metadata. This metadata 
		includes the split text's token sequence, index (with respect
		to the input text), and length of the text split for each split
		in the text.
	'''
	# Check that the splitters is non-empty.
	assert len(splitters) >= 1, "Expected high_level_split() argument 'splitters' to be populated"
	
	# Check the "top"/"first" splitter. Make sure that it is for
	# splitting the text at the paragraph level.
	valid_splitters = ["\n\n", "\n"]
	splitters_copy = copy.deepcopy(splitters)
	splitter = splitters_copy.pop(0)
	assert splitter in valid_splitters, "Expected first element for high_level_split() argument 'splitter' to be either '\\n\\n' or '\\n'"

	# Initialize the metadata list.
	metadata = []

	# Split the text.
	text_splits = text.split(splitter)

	# Iterate through the list 
	for split in text_splits:
		# Skip the split if it is an empty string.
		if split == "":
			continue

		# Get the split metadata (index with respect to original text 
		# plus offset and split length).
		split_idx = text.index(split) + offset
		split_len = len(split)

		# Tokenize the split.
		tokens = tokenizer.encode(split, add_special_tokens=False)

		if len(tokens) <= context_length:
			# If the token sequence is less than or equal to the 
			# context length, tokenize the text split again (this time
			# with padding), and add the entry to the metadata.
			tokens = tokenizer.encode(
				split, 
				add_special_tokens=False, 
				padding="max_length"
			)
			metadata.append({
				"tokens": tokens,
				"text_idx": split_idx,
				"text_len": split_len
			})
		else:
			# If the token sequence is greater than the context length,
			# pass the text over to the next splitter. Check the next
			# splitter and use the appropriate function.
			next_splitter = splitters_copy[0]
			if next_splitter in valid_splitters:
				metadata += high_level_split(
					split, 
					split_idx, 
					tokenizer, 
					context_length, 
					splitters_copy
				)
			else:
				metadata += low_level_split(
					split, 
					split_idx, 
					tokenizer, 
					context_length, 
					splitters_copy
				)

	# Return the metadata.
	return metadata


def low_level_split(
		text: str, 
		offset: int, 
		tokenizer: AutoTokenizer, 
		context_length: int, 
		splitters: List[str]
) -> List[Dict]:
	'''
	(Recursively) split the text into words or characters and extract
		the metadata from the text slices of the input text. If the 
		splits are too large, recursively call the function until the 
		text becomes manageable.
	@param: text (str), the text that is to be processed for storing to
		vector database.
	@param: offset (int), the index of the input text with respect to
		the original text.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: context_length (int), the maximum number of tokens 
		supported by the model. This helps us chunk the text if the 
		tokenized output is "too long".
	@param: splitters (List[str]), the list of strings that will be 
		used to split the text. For this function, we expect the 
		"top-most" strings to be either in the set (" ", "").
	@return: returns a List[Dict] of the text metadata. This metadata 
		includes the split text's token sequence, index (with respect
		to the input text), and length of the text split for each split
		in the text.
	'''
	# Check that the splitters is non-empty.
	assert len(splitters) >= 1, "Expected low_level_split() argument 'splitters' to be populated"
	
	# Check the "top"/"first" splitter. Make sure that it is for
	# splitting the text at the paragraph level.
	valid_splitters = [" ", ""]
	splitters_copy = copy.deepcopy(splitters)	# deep copy because this variable is modified
	splitter = splitters_copy.pop(0)
	assert splitter in valid_splitters, "Expected first element for low_level_split() argument 'splitter' to be either ' ' or ''"

	# Initialize the metadata list.
	metadata = []

	# Initialize a boolean to determine if the function needs to use
	# the next splitter in the recursive call or stick with the current
	# one. Initialize to True.
	use_next_spitter = True

	# Split the text.
	if splitter != "":
		# Split text "normally" (splitter is not an empty string "").
		text_splits = text.split(splitter)
	else:
		# Split text here if the splitter is "". The empty string "" is
		# not recognized as a valid text separator.
		text_splits = list(text)

	# Aggregate the splits according to the splitter. Current
	# aggregation strategy is to chunk the splits by half.
	half_len = len(text_splits) // 2
	if half_len > 0:	# Same as len(text_splits) > 1
		# This aggregation only takes affect if the number of items
		# resulting from the split is more than 1. Otherwise, there is
		# no need to aggregate.
		text_splits = [
			splitter.join(text_splits[:half_len]),
			splitter.join(text_splits[half_len:]),
		]

		# Flip boolean to False while the split list is still longer
		# than one item.
		use_next_spitter = False

	# Iterate through the list 
	for split in text_splits:
		# Skip the split if it is an empty string.
		if split == "":
			continue

		# Get the split metadata (index with respect to original text 
		# plus offset and split length).
		split_idx = text.index(split) + offset
		split_len = len(split)

		# Tokenize the split.
		tokens = tokenizer.encode(split, add_special_tokens=False)

		if len(tokens) <= context_length:
			# If the token sequence is less than or equal to the 
			# context length, tokenize the text split again (this time
			# with padding), and add the entry to the metadata.
			tokens = tokenizer.encode(
				split, 
				add_special_tokens=False, 
				padding="max_length"
			)
			metadata.append({
				"tokens": tokens,
				"text_idx": split_idx,
				"text_len": split_len
			})
		else:
			# If the token sequence is greater than the context length,
			# pass the text over to the next splitter. Since we are
			# already on the low level split function, we'll just
			# recursively call the function again.
			if not use_next_spitter:
				# If the boolean around using the next splitter is
				# False, re-insert the current splitter to the
				# beginning of the splitters list before it is passed
				# down to the recursive function call.
				splitters_copy.insert(0, splitter)

			metadata += low_level_split(
				split, 
				split_idx, 
				tokenizer, 
				context_length, 
				splitters_copy
			)

	# Return the metadata.
	return metadata


def load_model(
		# config: Dict, 
		model_config: Dict[str, Union[str, int]],
		model_name: str,
		device="cpu"
) -> Tuple[AutoTokenizer, AutoModel]:
	'''
	Load the tokenizer and model. Download them if they're not found 
		locally.
	@param: config (Dict), the configuration JSON. This will specify
		the model and its path attributes.
	@param: device (str), tells where to map the model. Default is 
		"cpu".
	@return: returns the tokenizer and model for embedding the text.
	'''
	# Check for the local copy of the model. If the model doesn't have
	# a local copy (the path doesn't exist), download it.
	# model_name = config["vector-search_config"]["model"]
	# model_config = config["models"][model_name]
	model_path = model_config["storage_dir"]
	
	# Check for path and that path is a directory. Make it if either is
	# not true.
	if not os.path.exists(model_path) or not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)

	# Check for path the be populated with files (weak check). Download
	# the tokenizer and model and clean up files once done.
	if len(os.listdir(model_path)) == 0:
		print(f"Model {model_name} needs to be downloaded.")

		# Check for internet connection (also checks to see that
		# huggingface is online as well). Exit if fails.
		response = requests.get("https://huggingface.co/")
		if response.status_code != 200:
			print(f"Request to huggingface.co returned unexpected status code: {response.status_code}")
			print(f"Unable to download {model_name} model.")
			exit(1)

		# Create cache path folders.
		cache_path = model_config["cache_dir"]
		os.makedirs(cache_path, exist_ok=True)
		os.makedirs(model_path, exist_ok=True)

		# Load tokenizer and model.
		model_id = model_config["model_id"]
		tokenizer = AutoTokenizer.from_pretrained(
			model_id, cache_dir=cache_path, device_map=device
		)
		model = AutoModel.from_pretrained(
			model_id, cache_dir=cache_path, device_map=device
		)

		# Save the tokenizer and model to the save path.
		tokenizer.save_pretrained(model_path)
		model.save_pretrained(model_path)

		# Delete the cache.
		shutil.rmtree(cache_path)
	
	# Load the tokenizer and model.
	tokenizer = AutoTokenizer.from_pretrained(
		model_path, device_map=device
	)
	model = AutoModel.from_pretrained(
		model_path, device_map=device
	)

	# Return the tokenizer and model.
	return tokenizer, model


def embed_docs(
		db: DBConnection, 
		config: Dict,
		model_name: str, 
		model_config: Dict[str, Union[str, int]], 
		tokenizer: AutoTokenizer, 
		model: AutoModel, 
		data: Dataset, 
		device: str = "cpu"
) -> None:
	# Initialize the table(s).
	dims = model_config["dims"]
	full_prec_table_name = f"{model_name}_fp32"
	binary_prec_table_name = f"{model_name}_binary"
	table_names = db.table_names()
	if full_prec_table_name not in table_names:
		schema = pa.schema([
			pa.field("wikidata_id", pa.string()),
			pa.field("text_idx", pa.int32()),
			pa.field("text_len", pa.int32()),
			pa.field("vector_full", pa.list_(pa.float32(), dims))
		])
		db.create_table(full_prec_table_name, schema=schema)

	if binary_prec_table_name not in table_names:
		dim_bytes = math.ceil(dims / 8)
		schema = pa.schema([
			pa.field("wikidata_id", pa.string()),
			pa.field("text_idx", pa.int32()),
			pa.field("text_len", pa.int32()),
			pa.field("vector_binary", pa.list_(pa.uint8(), dim_bytes))
		])
		db.create_table(binary_prec_table_name, schema=schema)

	full_prec_table = db.open_table(full_prec_table_name)
	binary_prec_table = db.open_table(binary_prec_table_name)

	# Iterate through the document ids.
	for entry in tqdm(data, desc="Embedding"):
		article_id, article_text = entry["wikidata_id"], entry["cleaned_text"]

		# Preprocess text (chunk it) for embedding.
		chunk_metadata = vector_preprocessing(
			article_text, config, model_config, tokenizer
		)

		# Embed each chunk and update the metadata.
		for idx, chunk in enumerate(chunk_metadata):
			# Update/add the metadata for the source filename
			# and article SHA1.
			chunk.update(
				{"wikidata_id": article_id}
			)

			# Get original text chunk from text.
			text_idx = chunk["text_idx"]
			text_len = chunk["text_len"]
			# text_chunk = article_text[text_idx: text_idx + text_len]
			text_chunk = chunk["tokens"]
			chunk.update(
				{"text_idx": text_idx, "text_len": text_len}
			)

			# Embed the text chunk.
			embeddings = embed_text(
				text_chunk, tokenizer, model, device, to_binary=True
			)
			embedding, binary_embedding = embeddings
			del chunk["tokens"]

			# NOTE:
			# Originally I had embeddings stored into the metadata
			# dictionary under the "embedding", key but lancddb
			# requires the embedding data be under the "vector"
			# name.

			# Update the chunk dictionary with the embedding
			# and set the value of that chunk in the metadata
			# list to the (updated) chunk.
			# chunk.update({"embedding": embedding})
			# chunk.update({"vector": embedding})
			chunk.update({
				"vector_full": embedding,
				"vector_binary": binary_embedding,
			})
			chunk_metadata[idx] = chunk

		# Add chunk metadata to the vector database. Should be on
		# "append" mode by default.
		# table.add(chunk_metadata, mode="append")
		full_prec_table.add(
			[
				{
					k: v for k, v in chunk.items() 
					if "binary" not in k
				} for chunk in chunk_metadata
			], 
			mode="append"
		)
		binary_prec_table.add(
			[
				{
					k: v for k, v in chunk.items() 
					if "full" not in k
				} for chunk in chunk_metadata
			], 
			mode="append"
		)

		# Cleanup artifacts.
		full_prec_table.optimize(
			cleanup_older_than=timedelta(seconds=30)
		)
		full_prec_table.cleanup_old_versions(
			older_than=timedelta(seconds=30)
		)
		binary_prec_table.optimize(
			cleanup_older_than=timedelta(seconds=30)
		)
		binary_prec_table.cleanup_old_versions(
			older_than=timedelta(seconds=30)
		)


def embed_text(
		text: Union[str, List[int]], 
		tokenizer: AutoTokenizer, 
		model: AutoModel, 
		device: str = "cpu",
		to_binary: bool = False
	) -> Tuple[np.array]:
	# Disable gradients.
	with torch.no_grad():
		if isinstance(text, str):
			# Pass original text chunk to tokenizer. Ensure the data is
			# passed to the appropriate (hardware) device.
			output = model(
				**tokenizer(
					text,
					add_special_tokens=False,
					padding="max_length",
					return_tensors="pt"
				).to(device)
			)
		elif isinstance(text, list) and all(isinstance(token, int) for token in text):
			input_ids = torch.tensor([text]).to(device)
			attention_mask = torch.tensor(
				[get_attention_mask(text, tokenizer.pad_token_id)]
			).to(device)
			output = model(
				input_ids=input_ids, attention_mask=attention_mask
			)
		else:
			raise ValueError(f"Expected text to be either string or List[int]. Recieved {type(text)}")

		# Compute the embedding by taking the mean of the last 
		# hidden state tensor across the seq_len axis.
		embedding = output[0].mean(dim=1)

		# Apply the following transformations to allow the
		# embedding to be compatible with being stored in the
		# vector DB (lancedb):
		#	1) Send the embedding to CPU (if it's not already
		#		there)
		#	2) Convert the embedding to numpy and flatten the
		# 		embedding to a 1D array
		embedding = embedding.to("cpu")
		embedding = embedding.numpy()[0]

		# Generate binary embeddings if specified.
		if to_binary:
			binary_embedding = (embedding > 0).astype(np.uint8)
			binary_embedding = np.packbits(binary_embedding, axis=-1)
			return (embedding, binary_embedding)
	
	# Return the embedding.
	return (embedding)


def get_attention_mask(tokens: List[int], pad_token_id: int) -> List[int]:
	return [0 if t == pad_token_id else 1 for t in tokens]


def clean_text(text: str) -> str:
	cleaned_text = text.replace("_START_ARTICLE_", "\n")\
		.replace("_START_SECTION_", "\n")\
		.replace("_START_PARAGRAPH_", "\n")
	
	return cleaned_text


def load_data() -> Dataset:
	# Files.
	dataset_name = "google/wiki40b"
	folder = dataset_name.replace("/", "_")
	splits = ["train", "test", "validation"]
	paths = [os.path.join(folder, split) for split in splits]
	
	# Load and return the dataset.
	data = concatenate_datasets([
		load_from_disk(path) for path in paths
	])
	return data


def main():
	# Files.
	dataset_name = "google/wiki40b"
	dataset_lang = "en"
	folder = dataset_name.replace("/", "_")
	cache_dir = folder + "_tmp"
	splits = ["train", "test", "validation"]

	# Check for the dataset already being downloaded.
	if not os.path.exists(folder) or len(os.listdir(folder)):
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
	data = load_data()
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
		model_config = model_configs[model_name]

		# Load model and tokenizer.
		tokenizer, model = load_model(model_config, model_name, device)

		# Embed data to database.
		embed_docs(
			db, config, model_name, model_config, tokenizer, model, data, device
		)
	
	# Exit the program.
	exit(0)

if __name__ == '__main__':
	main()