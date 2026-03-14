# Vector Text Embeddings Benchmark


The goal of this repo is to provide people with a way to benchmark the performance of language models for embedding text in RAG applications.


### Notes

 - While HKU nlp are included in the models section of the `config.json`, I had trouble trying to get just the embeddings from that model. The core model is an encoder-decoder model (similar to T5) while all other models were encoder-only models. As such, they're not included in the results. 
     - That being said, if anyone is able to bring them online and part of the evaluation, please be my guest and let me know how they perform.
 - The amount of storage required to run these experiments is not trivial.
     - For running this experiment on 1,000 articles from the dataset, please set aside around 40 GB of disk space.
     - The core dataset (`google wiki40b english`) requires 10 GB of storage.
     - The vector embeddings (for all reported models, across fp32 and binary precision, for all 1,000 articles in english) requires around 20 GB of storage.
     - The remaining storage budget is consumed via the models that are downloaded and stored.
 - This experiment does not consider multilingual data or models, only english.
     - Feel free to use this repo as a template to evaluate those kinds of datasets and models.
 - In order to generate queries that were relevant to the sampled passages, we used Flan-T5.
     - Feel free to swap this out with your favorite models.
     - Support for popular LLMs (i.e. Llama) is included but not refined/tested.
     - Also feel free to adjust the prompt (or other arguments to the model) as needed.
 - General results:
     - The best performing combination was the roberta larger (v1) model from sentence transformers using full fp32 precision and cosine distance.
     - The utilization of the minilm models from sentence transformers with binary embeddings (hamming distance) is the best performing combination in general.
         - Roberta variants also performed well (also with binary embeddings over fp32 precision).
     - Using binary embeddings reduces storage requirements signficantly.
     - Classic bert models performed the worst (regardless of precision or distance metric).
 - To replicate the work:
     - Build environment:
         - Use conda (or similar virtual environment software) to set up a virtual environment.
             - `conda env create -f env.yml` or `conda env create -f env-cuda.yml` if you have an Nvidia GPU with CUDA 11 or 12.
             - `conda activate text-emb`
     - Download the data:
         - `python download.py`
     - Embed the data to vector database:
         - `python embed.py --subset 1000 --batch_size 128`
         - You can also specify which split (`train`, `test`, `validation`) you want to embed with the `--split` argument. Default is that it looks at all splits at once.
         - The `--batch_size` argument tells you how big of a batch size to use when embedding the data with the model. Default batch size is 1.
     - Run the benchmark script:
         - `python benchmarks.py --subset 1000 --sample 3`
         - Just like the `embed.py` script, you can also specify which split you want to load with the `--split` argument. You should specify the same `--split` and `--subset` arguments that you used to embed the data to keep things consistent, otherwise you run the risk of trying to query data that may or may not be in the database.
         - The `--sample` argument tells you how many articles to sample and generate queries from. Default is 3 but you can use more or less as desired.


### References

 - huggingface
     - [wiki40b dataset](https://huggingface.co/datasets/google/wiki40b)
     - [blog on embedding quantization](https://huggingface.co/blog/embedding-quantization)