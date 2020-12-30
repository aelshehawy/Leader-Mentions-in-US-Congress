# Leader-Mentions-in-US-Congress

## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Usage](#usage)

## General info
This project is implemented in Python3. It provides:

* A: a script for extracting spelling variants for person names from political speeches
* B: a script for counting mentions of political leaders and countries in political speeches 
  (also using the name variants from A to increase recall)
 
	
## Requirements

The script requires the following Python libraries: 
* pandas 
* sklearn
* Levenshtein


Information on how to install:

* pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
* sklearn: https://scikit-learn.org/stable/install.html
* Levenshtein: https://github.com/ztane/python-Levenshtein

	
## Usage

### A: Extraction of spelling variants for person names

To extract name variants, run:

	python extract_name_variants.py 


### B: Counting leader and country mentions in political speeches

To count leader and country mentions, run:

	python count_mentions.py


