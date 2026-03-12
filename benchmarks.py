# benchmark.py
# Run a comparative benchmark for each of the models at different 
# quantization.
# Python 3.9
# Windwos/MacOS/Linux


import argparse
import copy
import json
import os
import random
import re
import shutil
from typing import Any, Dict, List

from datasets import Dataset, load_dataset
import lancedb
from lancedb import table
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

from embed import load_data, load_model, clean_text
from embed import batch_embed_text, vector_preprocessing


seed = 1234
random.seed(seed)


def calc_rr(pos):
	'''
	Calculate the reciprocal rank.
	'''
	return 1.0 / (pos + 1) if pos >= 0 else 0.0


def get_passages(
		data: Dataset, 
		text_id: str, 
		substr_idx: int, 
		substr_len: int
	) -> str:
	row = data.filter(lambda x: x['wikidata_id'] == text_id)

	if len(row) == 0:
		return "ID not found."
	
	# Extract the full text from the first match
	full_text = row[0]['text']
	
	# Return the substring based on index and length.
	return clean_text(full_text)[substr_idx : substr_idx + substr_len]


def perform_search(
		table: table, 
		query_vector: List[float], 
		metric: str = "l2", 
		limit: int = 5
	) -> List[Dict[str, Any]]:
	'''
	Abstracted search function that accepts any supported metric.
	'''
	# LanceDB uses .metric() or .distance_type() in the query builder
	results = (
		table.search(query_vector)
		.metric(metric) 
		.limit(limit)
		.to_list()
	)
	return results


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
		"--sample",
		type=int,
		default=3,
		help="How many articles should be sampled from the dataset in order to generate questions. Default is 3."
	)
	args = parser.parse_args()

	###################################################################
	# DATA SETUP
	###################################################################

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

	###################################################################
	# CONFIG SETUP
	###################################################################

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
	local_save_detected = os.path.exists(gen_model_save)
	if not local_save_detected:
		gen_tokenizer = AutoTokenizer.from_pretrained(
			gen_model_name,
			cache_dir=gen_model_tmp
		)
		gen_tokenizer.save_pretrained(gen_model_save)
		shutil.rmtree(gen_model_tmp)

	gen_tokenizer = AutoTokenizer.from_pretrained(
		gen_model_save
	)

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
	sample_size = args.sample
	assert sample_size <= len(data) and sample_size > 0, \
		f"Expected argument --sample to be between {1} and {len(data)}, inclusive. Recieved {sample_size}"

	query_metadata = list()
	for i in range(min(sample_size, len(data))):
		# Get a random article.
		random_article = data[random.randint(0, len(data) - 1)]
		article_id = random_article["wikidata_id"]
		context = random_article["cleaned_text"]
		
		# To keep the context manageable, let's take a snippet by 
		# sampling a paragraph from the original text.
		context_snippet = get_random_paragraph_from_article(context)
		print(f"\nArticle Snippet ({i + 1}):\n{context_snippet}...\n")
		
		# Generate the question.
		question = generate_question_from_context(
			context_snippet, gen_model, gen_tokenizer, device
		)

		query_metadata.append(
			(article_id, context, context_snippet, question)
		)
		print(f"Generated Question: {question}")
		print("--------------------------------------------------\n")

	###################################################################
	# BENCHMARK (QUERYING LOOP)
	###################################################################

	# Iteration loop:
	# For _, _, question in article_snippet_question:
	#   For model_name in models:
	#      For precision in precision_list:
	#         if precision == binary distance metrics = [hamming] else [l2, cosine, dot]
	#         For distance_metric in distance_metrics:
	#            For n in top_n:
	#               Run search and check if question returns the correct context in 
	# Resulting table (pandas dataframe).
	# article_id | context | context_snippet | question | model_name | precision | distance_metric | top_n | article_in_top_n | article_position | passage_in_top_n | passage_position

	# Loop variables.
	precision_list = ["fp32", "binary"]
	distance_metrics = gen_config["distance_metrics"]
	top_n = gen_config["top_n"]
	max_top_n = max(top_n)
	analysis_results = list()

	# Iterate through the question.
	for article_id, _, context_snippet, question in query_metadata:
		# Iterate through the models.
		for model_name in model_names:
			# TODO:
			# Fix embedding call for hkunlp models. They are 
			# encoder-decoder models similar to T5, so calling the 
			# model itself is not sufficient. You'd have to do 
			# something like model.encoder(**inputs).

			# Skip hkunlp models (see above TODO note).
			if "hkunlp" in model_name:
				continue

			# Load model config.
			model_config = model_configs[model_name]

			# Load model and tokenizer.
			tokenizer, model = load_model(
				model_config, model_name, device
			)

			# Preprocess the question for this particular model.
			metadata = vector_preprocessing(
				question, config, model_config, tokenizer, device
			)

			# Slice only the first object since we're asking one query 
			# at a time.
			query_chunk = metadata[0]

			# Pad if necessary.
			length_diff = model_config["max_tokens"] - len(query_chunk["tokens"])
			if length_diff != 0:
				tokens = query_chunk["tokens"]
				tokens.extend([tokenizer.pad_token_id] * length_diff)
				query_chunk.update({"tokens": tokens})
			
			# Embed the question.
			query_tokens = [query_chunk["tokens"]]
			embeddings, binary_embeddings = batch_embed_text(
				query_tokens, tokenizer, model, device, True
			)

			# Iterate through the precision levels. Load table
			# accordingly.
			for precision in precision_list:
				# Load metrics based on precision.
				if precision == "binary":
					distance_metrics_set = ["hamming"]
				else:
					distance_metrics_set = copy.deepcopy(distance_metrics)
					distance_metrics_set.remove("hamming")

				# Load table.
				table_name = f"{model_name}_{precision}"
				table = db.open_table(table_name)

				# Set the query vector.
				query_vector = binary_embeddings if precision == "binary" else embeddings

				# Iterate through each metric and perform the search.
				for metric in distance_metrics_set:
					# Run the search.
					results = perform_search(
						table, query_vector, metric=metric, limit=max_top_n
					)

					results_ids = [result["wikidata_id"] for result in results]
					result_strings = [
						get_passages(
							data, 
							result["wikidata_id"], 
							result["text_idx"], 
							result["text_len"]
						)
						for result in results
					]

					for top_val in sorted(top_n):
						# Check if article id in the results list.
						article_in_top_n = article_id in results_ids[:top_val]
						article_position = results_ids[:top_val].index(article_id) if article_in_top_n else -1

						# Check if context snippet used to generate the 
						# question is in the results list.
						passage_in_top_n = False
						passage_position = -1
						for idx, string in enumerate(result_strings[:top_val]):
							if context_snippet in string:
								passage_in_top_n = True
								passage_position = idx
								break

						# Build out the analysis data.
						analysis_results.append({
							"article_id": article_id,
							"context": context_snippet,
							"question": question,
							"model_name": model_name,
							"precision": precision,
							"distance_metric": metric,
							"top_n": top_val,
							"article_in_top_n": article_in_top_n,
							"article_position": article_position,
							"passage_in_top_n": passage_in_top_n,
							"passage_position": passage_position
						})

	###################################################################
	# SAVING BENCHMARK DATA
	###################################################################
	# Convert the analysis results to a dataframe.
	analysis_df = pd.DataFrame(analysis_results)
	print(analysis_df.head())
	analysis_df.to_csv("benchmark_results.csv", sep="|")

	###################################################################
	# PLOTTING BENCHMARK DATA
	###################################################################
	analysis_df['reciprocal_rank'] = analysis_df['passage_position'].apply(calc_rr)

	# Grouping data for generic Recall analysis across Top N
	grouped = analysis_df.groupby(['model_name', 'distance_metric', 'precision', 'top_n']).agg(
		passage_recall=('passage_in_top_n', 'mean')
	).reset_index()
	grouped['configuration'] = grouped['model_name'] + " (" + grouped['precision'] + ", " + grouped['distance_metric'] + ")"

	# Filter for the highest Top N (50) to evaluate overall rank quality (MRR)
	df_final = analysis_df[analysis_df['top_n'] == analysis_df['top_n'].max()].copy()
	mrr_df = df_final.groupby(
		['model_name', 'precision', 'distance_metric']
	)['reciprocal_rank'].mean().reset_index()
	mrr_df['configuration'] = mrr_df['model_name'] + " (" + mrr_df['precision'] + ", " + mrr_df['distance_metric'] + ")"

	# A. Recall Curve (Efficiency vs. Depth)
	plt.figure(figsize=(14, 8))
	sns.lineplot(data=grouped, x='top_n', y='passage_recall', hue='model_name', marker='o')
	plt.title('Recall@N Comparison across Models')
	plt.ylabel('Recall Rate')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.grid(True, linestyle='--', alpha=0.3)
	plt.tight_layout()
	plt.savefig('recall_curve.png')

	# B. MRR Leaderboard (Overall Search Quality)
	mrr_sorted = mrr_df.sort_values(by='reciprocal_rank', ascending=False)
	plt.figure(figsize=(12, 10))
	sns.barplot(data=mrr_sorted, x='reciprocal_rank', y='configuration', palette='magma')
	plt.title('Mean Reciprocal Rank (MRR) by Model Configuration')
	plt.xlabel('MRR (Higher is Better)')
	plt.grid(axis='x', linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.savefig('mrr_comparison.png')

	# C. Model Robustness (Metric sensitivity)
	plt.figure(figsize=(12, 8))
	model_order = mrr_df.groupby('model_name')['reciprocal_rank'].median().sort_values(ascending=False).index
	sns.boxplot(data=mrr_df, y='model_name', x='reciprocal_rank', order=model_order, palette='vlag')
	sns.stripplot(data=mrr_df, y='model_name', x='reciprocal_rank', order=model_order, color='black', alpha=0.3)
	plt.title('Model Robustness: Distribution of MRR across Metrics/Precisions')
	plt.tight_layout()
	plt.savefig('model_robustness.png')

	# D. Precision Impact Heatmap (FP32 vs. Binary)
	precision_pivot = mrr_df.groupby(['model_name', 'precision'])['reciprocal_rank'].mean().unstack()
	precision_pivot = precision_pivot.sort_values(by='fp32', ascending=False)
	plt.figure(figsize=(10, 8))
	sns.heatmap(precision_pivot[['fp32', 'binary']], annot=True, cmap='RdYlGn', fmt='.3f')
	plt.title('Performance Comparison: FP32 vs Binary Average MRR')
	plt.tight_layout()
	plt.savefig('precision_heatmap.png')

	# Keep only the best-performing metric for each (Model, Precision) combination
	leaderboard = mrr_df.sort_values(['model_name', 'reciprocal_rank'], ascending=[True, False])
	leaderboard = leaderboard.groupby(['model_name', 'precision']).head(1).sort_values(by='reciprocal_rank', ascending=False)
	leaderboard.to_csv('model_leaderboard.csv', index=False)
	
	# Exit the program.
	exit(0)


if __name__ == '__main__': 
	main()