# download.py
# Download the dataset (wiki-40b english).
# Python 3.9
# Windwos/MacOS/Linux


import os
import shutil

from datasets import load_dataset


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

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()