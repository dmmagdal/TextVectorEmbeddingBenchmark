# benchmark.py
# Run a comparative benchmark for each of the models at different 
# quantization.
# Python 3.9
# Windwos/MacOS/Linux


import argparse
import json
import math
import os
import random
import re
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
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

from embed import load_data, load_model, clean_text


seed = 1234
random.seed(seed)


def generate_question_from_context(
    context: str, 
    model: AutoModelForSeq2SeqLM, 
    tokenizer: AutoTokenizer, 
    device: str
) -> str:
    '''
    Generates a question from a given context using a pre-loaded generative model.
    This example is tailored for a Seq2Seq model like Flan-T5.
    '''
    prompt = (
        "Given the following context, generate a question that can be "
        f"directly answered by it.\n\nContext: {context}\n\nQuestion:"
    )

    inputs = tokenizer(
		prompt, 
		return_tensors="pt", 
		max_length=512, 
		truncation=True
	).to(device)
    outputs = model.generate(
		**inputs, 
		max_new_tokens=1024, 
		num_beams=4, 
		early_stopping=True
	)
    question = tokenizer.decode(
		outputs[0], 
		skip_special_tokens=True
	)
    
    return question


def get_random_paragraph_from_article(article_text: str) -> str:
	'''
	Get a random paragraph from the wikipedia article.
	@param: article_text (str), the text of the wikipedia article.
	@return: returns a random, non-empty paragraph from that article.
	'''
	# Return the article text (empty string) if the article text is 
	# just an empty string.
	if article_text == "":
		return article_text
	
	# Split article text by paragraph and remove all empty string 
	# entries.
	article_text = article_text.replace("_NEWLINE_", "\n")
	split_text = [
		text.strip() 
		# for text in article_text.split("\n\n") # Better chances of longer passages
		for text in article_text.split("\n") # More "natural" but shorter passages (will impact performance)
		if text.strip()	
	]

	# Crude wiki markup cleanup.
	for idx, text in enumerate(split_text):
		text = re.sub(r"\{\{.*?\}\}", "", text)  # remove templates
		# text = re.sub(r"\[\[.*?:.*?\]\]", "", text)  # categories/files
		# text = re.sub(r"==.*?==", "", text)  # headers
		text = re.sub(r"<.*?>", "", text)  # html tags
		split_text[idx] = text

	# Compute weights as lengths of each paragraph.
	weights = []
	for text in split_text:
		weight = len(text)

		# If a paragraph (split along individual newline characters) is
		# too long (indicating some sort of table or other structure),
		# nullify the weight for that paragraph.
		# if len(text.split("\n")) > 5:
		# 	weight = math.ceil(weight * 0.01)

		# Append the weight to the list.
		weights.append(weight)

	# Randomly sample a paragraph from the split text. Weigh the 
	# choices by the length of the paragraph.
	return random.choices(
		split_text, 
		# weights=[len(text) for text in split_text],
		weights=weights,
		k=1
	)[0]


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
		data = data.select(range(min(args.subset, len(data))))

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

	# Load generative model.
	gen_config = config["test"]
	gen_model_name = gen_config["model"]
	gen_model_names = sorted(gen_config["models"])
	assert gen_model_name in gen_model_names, \
		f"Expected target 'model' from 'test' in config.json to have a valid model name ({', '.join(gen_model_names)}). Recieved '{gen_model_name}'."
	
	# For models like Flan-T5 (encoder-decoder), use AutoModelForSeq2SeqLM.
	# For decoder-only models like Llama 3, you would use AutoModelForCausalLM.
	# Note: Llama 3 is a gated model and requires authentication.
	print(f"Loading generative model {gen_model_name} for question generation...")
	gen_model_save = gen_model_name.replace("/", "_")
	gen_model_tmp = gen_model_save + "_tmp"
	gen_tokenizer = AutoTokenizer.from_pretrained(
		gen_model_name,
		cache_dir=gen_model_tmp
	)
	local_save_detected = os.path.exists(gen_model_save)
	if "t5" in gen_model_name:
		if not local_save_detected:
			gen_model = AutoModelForSeq2SeqLM.from_pretrained(
				gen_model_name,
				cache_dir=gen_model_tmp
			)
			gen_model.save_pretrained(gen_model_save)
			shutil.rmtree(gen_model_tmp)

		# Load saved model.
		gen_model = AutoModelForSeq2SeqLM.from_pretrained(
			gen_model_save
		).to(device)
	else:
		if not local_save_detected:
			gen_model = AutoModelForCausalLM.from_pretrained(
				gen_model_name,
				cache_dir=gen_model_tmp
			)
			gen_model.save_pretrained(gen_model_save)
			shutil.rmtree(gen_model_tmp)

		# Load saved model.
		gen_model = AutoModelForCausalLM.from_pretrained(
			gen_model_save
		).to(device)
	print(f"Loaded {gen_model_name} on {device}.")

	# Generate questions.
	print("\n--- Generating sample questions from the dataset ---")
	article_snippet_question = list()
	for i in range(min(3, len(data))): # Generate for 3 articles
		# Get a random article.
		random_article = data[random.randint(0, len(data) - 1)]
		context = random_article["cleaned_text"]
		
		# To keep the context manageable, let's take a snippet by 
		# sampling a paragraph from the original text.
		context_snippet = get_random_paragraph_from_article(context)
		print(f"\nArticle Snippet ({i + 1}):\n{context_snippet}...\n")
		
		# Generate the question.
		question = generate_question_from_context(
			context_snippet, gen_model, gen_tokenizer, device
		)

		article_snippet_question.append(
			(context, context_snippet, question)
		)
		print(f"Generated Question: {question}")
		print("--------------------------------------------------\n")
	
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