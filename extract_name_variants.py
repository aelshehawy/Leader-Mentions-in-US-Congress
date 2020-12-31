import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import Levenshtein as ls
import csv, json
import glob, time
from os import path
import re
import sys


"""
Read a csv file with person names (first_name(s) last_name: William Ofori Atta)
and election dates and return a dictionary with all names listed for each date
in the file.
"""
def read_names(infile):
	df = pd.read_csv(infile, header=0, sep='\t')
	name_dic = {}
	for idx, row in df.iterrows():
		if row.date not in name_dic:
			name_dic[str(row.date)] = []
		name_dic[str(row.date)].append(row.leader_name)
	return name_dic



"""
Extract name variants for the given names from the speeches
that have a similarity below the threshold.
"""
def compute_name_variants(dic, full_name, df, threshold):
    
    # Extract speech text from the dataframe
	text = []
	for idx, row in df.iterrows():
		text.append(row.speech)
    
    # Extract the vocabulary (unigrams, bigrams, trigrams)
    # => keeps upper-cased letters but ignores punctuation.
	vectorizer = CountVectorizer(lowercase=False, ngram_range=(1,3))
	X = vectorizer.fit_transform(text)
	vocab = vectorizer.get_feature_names()

    # Now use python-Levenshtein to compute the minimal edit distance between
    # a) the full name and b) the last name and all strings (unigrams, bigrams
    # and trigrams) in the vocabulary.
	new_variants = []
	for name in dic[full_name]:
		for v in vocab:
			sim = ls.ratio(name, v) 
			if sim > threshold and sim <1.0:
				if (v, sim) not in new_variants:
					new_variants.append((v, sim))
	return new_variants




if __name__ == "__main__":  
	# Input file with leader names and dates (year-month, e.g. 198612).
	# The dates specify the files from which we extract the name variants
	# (e.g. for 198612, we extract variants from the speeches in the 
	# corpus file <corpus/198612.csv>).
	name_file    = 'leader_names.csv'  

	# Directory with speeches from which we extract the name variants.
	# Input files have file names yyyymm.csv (e.g. 198612.csv). 
	# Each csv file has a header line <,speech>.
	corpus_path = 'corpus/'

	# Output file in json format, with extracted name variants and 
	# edit distance scores.
	results_file = 'name_variants.json'

	threshold = 0.85    # => for higher recall, decrease threshold
						# => for higher precision, increase threshold

	#####
	# First, we read in the leader names with election dates for
	# which we want to extract name variants. We use the election
	# dates to filter the corpus input used to extract name variants
	# (we only look for name variants in corpus files that are 
	# relevant for the elections we are interested in). 
	name_dic = read_names(name_file)

	
	old_variants = {}  
	new_variants = {}


	for date in name_dic:
		for full_name in name_dic[date]: 
			last_name = full_name.split(" ")[-1]
			# start looking for spelling variants of full_name and last_name
			# (e.g.: "Krishna Prasad Bhattarai", "Bhattarai")
			if full_name not in old_variants:
				old_variants[full_name] = [full_name, last_name]
				new_variants[full_name] = {}


		#####
		# Next, we read in the corpus file for this name item 
		# (from csv files with speeches in the directory <corpus_path> 
		# that were given in the time period specified by <date>).
		csv_path = corpus_path + date + '.csv'

		# Check if the file exists  
		if path.exists(csv_path): 
			try:
				df = pd.read_csv(csv_path)
			except IOError as e:
				print("I/O error({0}): {1}".format(e.errno, e.strerror))
			except:
				print("Unexpected error:", sys.exc_info()[0], csv_path)

			print("\tExtract name variants for: ", full_name, "\tfrom: ", date)
			new = compute_name_variants(old_variants, full_name, df, threshold) 
			for next_var, score in new:
				if next_var not in new_variants[full_name]:
					new_variants[full_name][next_var] = score
		else:
			print("ERROR: file path doesn't exist: ", csv_path)
		

	# Write extracted spelling variants to json file
	with open(results_file, 'w', encoding='utf-8') as f:
		json.dump(new_variants, f, ensure_ascii=False, indent=3)
