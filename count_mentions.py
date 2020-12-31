import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import Levenshtein as ls
from spacy.tokens import Span
from spacy.tokens import Token
import csv
import json
import glob, time
from os import path
import re
from tinydb import TinyDB, Query
import sys


"""
We have a subset of speeches where we found string matches for mentions of country names.
We now need to validate those matches, filter out noise and count the remaining mentions:
    - filter out matches listed under "no" in the dictionary (e.g., when looking for mentions of Korea/South Korea, we want to exclude mentions of "North Korea" and "Democratic People's Republic of Korea")
"""
def validate_counts(dic, country_dic): 
    for eid in dic:
        for name in dic[eid]:
            for date in dic[eid][name]:
                country = dic[eid][name][date]['country']
                offsets = []
                counts  = 0
                for idx, speech in dic[eid][name][date]['speeches']:
                    offsets_yes = []  
                    offsets_no = [] 
                    # Get the offsets for mentions listed in the "yes" list (offsets_yes) 
                    # and those listed in the "no" list (offsets_no).
                    offsets_yes += [([(m.start(0), m.end(0)) for m in re.finditer(r'\b'+var+r'\b', speech)], var) for var in country_dic[country]['yes']]
                    offsets_no  += [([(m.start(0), m.end(0)) for m in re.finditer(r'\b'+var+r'\b', speech)], var) for var in country_dic[country]['no']]                 
                    # Now we need to check if any of the instances in the offsets_no list 
                    # is overlapping with the ones in offsets_yes. If true, remove those
                    # mentions from the offsets_yes list.        
                    offsets += remove_false_pos(offsets_yes, offsets_no, speech)

                if len(offsets) >0:
                    counts += len(offsets) 
                dic[eid][name][date]['counts'] = counts  

    return dic            


""" 
Read a dictionary of country names and terms of interest
and return as dictionary
"""
def read_country_dict(country_dict_path): 
    with open(country_dict_path) as inf:
        dic = json.load(inf)  
    return dic


"""
Write results for country counts to file
"""
def write_counts_to_file(file_path, dic):
    with open(file_path, 'w') as out:
        for eid in dic:
            for name in dic[eid]:
                for date in dic[eid][name]:
                    out.write(eid + "\t" + date + "\t" + name + "\t" + dic[eid][name][date]['country'] + "\t" + str(dic[eid][name][date]['counts']) + "\n")


""" 
Remove all instances from a list that overlap with each other (e.g.: Korea, South Korea).
"""
def remove_duplicates(list1, speech):
    delete = []
    for i in range(len(list1)):
        start1, end1 = list1[i]
        w1 = speech[start1:end1]
        for j in range(i+1, len(list1)):
            start2, end2 = list1[j]
            w2 = speech[start2:end2] 
            # check if the match from list1 overlaps with another match in the list
            # case1: w1 is a substring of w2
            if start1 >= start2 and end1 <= end2: 
                delete.append((start1, end1))
            # case 2: w1 overlaps but starts and ends before w2
            elif start1 <= start2 and end1 >= start2:
                delete.append((start1, end1))
            # case 3: w1 overlaps but starts and ends after w2
            elif start1 <= end2 and end1 >= end2:
                delete.append((start1, end1))            
    # now remove all matches in delete and return the filtered list
    for start, end in list1:
        if (start, end) in delete:
            list1.remove((start, end))
    return list1




""" 
Remove all instances from the yes_list (listed in the country dictionary) 
that overlap with matches from the no_list (e.g.: Korea, North Korea).
"""
def remove_false_positives(list1, list2, speech):
    delete = []
    for i in range(len(list1)):
        start1, end1 = list1[i]
        w1 = speech[start1:end1]
        for j in range(len(list2)):
            start2, end2 = list2[j]
            w2 = speech[start2:end2] 
            # check if the match from list1 overlaps with a match in list2
            # case1: w1 is a substring of w2
            if start1 >= start2 and end1 <= end2: 
                delete.append((start1, end1))
            # case 2: w1 overlaps but starts and ends before w2
            elif start1 <= start2 and end1 >= start2:
                delete.append((start1, end1))
            # case 3: w1 overlaps but starts and ends after w2
            elif start1 <= end2 and end1 >= end2:
                delete.append((start1, end1))            
    # now remove all matches in delete and return the filtered list
    for start, end in list1:
        if (start, end) in delete:
            list1.remove((start, end))
    return list1



"""
Take a list with offsets for instances we want to consider for a given country name 
(e.g. mentions of Korea, South Korea for country=Korea) and a list with offsets for  
negative instances that we do not want to consider (e.g. North Korea for country=Korea). 
The two lists are specified in the json input file 'countries_yes_no.json'.
We now want to check if any of the instances in the offsets_no list is overlapping with 
the ones in offsets_yes. If that is the case, we remove those mentions from the offsets_yes 
list and return the filtered list.  
"""
def remove_false_pos(yes_list, no_list, speech):
    offsets = []
    look_up_yes = []  
    look_up_no  = []

    # get all start/end positions that should not be counted for yes list
    [look_up_yes.append((y_start, y_end)) for y_index_list, var in yes_list for (y_start, y_end) in y_index_list]
    # get all start/end positions that should not be counted for no list
    [look_up_no.append((n_start, n_end)) for n_index_list, var in no_list for (n_start, n_end) in n_index_list]    
    # first, remove duplicates from yes_list (e.g. Vietnam, Vietnamese)
    yes_list = remove_duplicates(look_up_yes, speech)

    # next, remove all instances from the yes_list that overlap with matches from the no_list
    if len(look_up_no) >0:
        yes_list = remove_false_positives(look_up_yes, look_up_no, speech)

    return yes_list



"""
We want to count all mentions of country names for each time period of interest 
(i.e., 3 months before/after an election). 
We iterate over all names in the name_dic and
    - select the speeches from csv files for the time periods we are interested in
    - and count the name mentions in the speeches 

Format name_dic:    
{
   "732-1985-0212-L1": {
      "Chun Doo-hwan": {
         "country": "South Korea",
         "date": [
            "198502",
            "198501",
            "198502",
            "198503"
         ]
      }
   },...
}

Format country_idc:
{
   "Congo": [
      "Congo",
      "West Congo",
      "Republic of the Congo"
   ],
	...
}
Format corpus files (corpus/yyyymm.csv):  speech_id,speech

Return dictionary with counts for each eid and name.
"""
def count_country_mentions_in_speeches(name_dic, text_snippet_path, country_dic):
	dic = {}    

	for eid in name_dic:
		for full_name in name_dic[eid]:
			dates = name_dic[eid][full_name]['date']
			country = name_dic[eid][full_name]['country'] 
			# Read all files included in dates (covering 3 months before/after election).
			for date in dates:
				# read the speech file for this election id / political leader / date  
				csv_path = corpus_path + date + '.csv'  

				if path.exists(csv_path):  
					try:
						df = pd.read_csv(csv_path)
					except IOError as e:
						print("I/O error({0}): {1}".format(e.errno, e.strerror))
					except:
						print("Unexpected error:", sys.exc_info()[0], csv_path)    

					# This shouldn't happen (all countries should be included in the dictionary)
					if country not in country_dic[country]["yes"]:
						# if not, add country to the list and notify user 
						country_dic[country]["yes"].append(country)

					for name_variant in country_dic[country]["yes"]:                
						hits = df[df['speech'].astype(str).str.contains(name_variant)]
						for idx, row in hits.iterrows():
							if eid not in dic:
								dic[eid] = {}
							if full_name not in dic[eid]:
								dic[eid][full_name] = {} 
							if date not in dic[eid][full_name]:
								dic[eid][full_name][date] = { 'country': country, 'speeches': []}

							if (row[0], row.speech) not in dic[eid][full_name][date]['speeches']:
								dic[eid][full_name][date]['speeches'].append((row[0], row.speech))

	# Now we have a dictionary with speeches that contain string matches for each country name for the period of interest. Now filter out false positives listed in the dictionary.
	dic = validate_counts(dic, country_dic)
	return dic



"""
Takes a python dictionary and a list of names
and adds new entries to the dictionary:
Format:     dic{ eid { name { 'country': country, 'date':[..., date]}}}

Example:
{'651-1945-0108-L1': 
    {'Mostafa El-Nahas': 
        {'country': 'Egypt', 
         'date': ['194411', '194412', '194501', '194502', '194503', '1945:4']
        }, 
     'Ahmad Mahir Pasha': 
        {'country': 'Egypt', 
          'date': ['194411', '194412', '194501', '194502', '194503', '194504']
        }
    }, ...
"""
def add_names_to_dic(dic, name, eid, country, date):

	if eid not in dic:
		dic[eid] = { name: {'country': country, 'date': [date] } }  
	elif name not in dic[eid]:
		dic[eid][name] = {'country': country, 'date': [date] } 
	else:
		# sanity check: dic country should be the same as country
		assert(dic[eid][name]['country'] == country) 
		dic[eid][name]['date'].append(date)
	return dic



"""
Read input file with leader names and dates (yyyymm; where the dates refer to the time period for which we want to search for leader mentions, i.e., 3 months before and after an election date).

Format: election_id	  leader_name   date   country

Return info as dictionary.

Format: 
{ name 
	{ 	'eid': 'election_id', 
		'date': yyyymm, 
		'country': 'country_name' }}} 
"""
def get_names_from_file(infile):
	dic = {}
	dates_of_interest = []
	df = pd.read_csv(infile, header=0, sep='\t')
	for idx, row in df.iterrows():
		eid = row.election_id
		date = str(row.date)
		name = row.leader_name
		country = row.country

		# create a new dictionary entry for each new electionid
		dic = add_names_to_dic(dic, name, eid, country, date)
		dates_of_interest.append(date)
	return dic, dates_of_interest




"""
Takes two name variants, two start offsets and two end offsets
and returns True if variant 1 is a substring of variant two
and comes from the same instances in the speech.
"""
def is_substring(n1, n2, s1, s2, e1, e2):
    
    if n1 in n2 and s1 >= s2 and e1 <= e2:
        return True
    return False



"""
Take a spacy doc representation of a speech and a filtered list of named entities:
Format: [('Vogel', 1464, 1469)]     =>  [(name_variant, start_offset, end_offset), (...)].
Return filtered list.

"""
def is_NER(full_name, doc, names, first_names): 

    # Get spacy NER annotations with indices and offsets.
    ents = [(ent, ent.start, ent.end, ent.label_, doc[ent.start].idx) for ent in doc.ents if ent.label_ == 'PERSON']
    filtered = []
    # Get token ids for items in names.
    for var, start, end in names:
        tok_ids = [i for i in range(len(doc)) if doc[i].idx == start]

        if len(tok_ids) >0:
            # We found the token id of the name mention in names.
        	# Now we check if the previous token is also part of the named entity string.
        	# If yes, then compare this string with full_name: 
        	# 	if the first name is part of the full name
        	#   => accept all instances of this name mention in this speech 
        	#      (one sense per speech heuristic: it is unlikely that the speech
        	#       mentions different leaders with the same name in the same speech).
            this_idx = tok_ids[0]
            prev_idx = this_idx -1
            # 380 is spacy's internal entity type for PERSON
            if doc[this_idx].ent_type != 380:
                continue      

            # Check for named entities of type PERSON (380): 
            # if the previous token is a named entity but is not part of the full_name 
            # => skip this entry.
            if doc[prev_idx].ent_type == 380 and doc[prev_idx].text not in full_name:
                continue
            else:
                filtered.append((var, start, end))

    return filtered



"""
Now we have a subset of speeches where we found string matches for name variants
for political actors. Next, we need to validate those matches and count them:
    - we filter out name lists (based on ratio of NE / word count in speech)
    - we filter out non-name mentions based on NER
    - if we have last-name mentions only, can we assure that they refer to the same name?
        (e.g. Hans-Jochen Vogel vs. Wolfgang Vogel vs. Herbert D. Vogel etc.)


Format dic:          
{'630-1979-0803-A1': {'Grand Ayatollah Khomeini': {'197908': {'speeches': [(960103975, 'Mr. President. on July...'), (...)]}}}}

Format variants:    
{ 'Hans-Jochen Vogel': ['Vogel', ...], 'Angela Merkel': [...] }
"""
def filter_and_count(dic, variants, first_names, result_file, context_file):
    # path to the retrained spacy NER model
    model_path = "spacy_model/en_core_web_lg_retrained/model-best"
    # load spacy model
    nlp = spacy.load(model_path)
    counts = {}

    for eid in dic:
        counts[eid] = {}
        for name in dic[eid]:
            counts[eid][name] = {}
            for date in dic[eid][name]:
                if date not in counts[eid][name]:
                    counts[eid][name][date] = {}
                counts[eid][name][date]['count'] = 0
                counts[eid][name][date]['mentions'] = []
                
                for idx, speech in dic[eid][name][date]['speeches']:
                    doc = nlp(speech) 

                    ### Filter out name lists:
                    #   - count number of PERSON tags in NER
                    #   - if ratio PERSON/token_count >0.08 => skip this instance
                    #	  (the value of 0.08 was determined empirically on our data;
                    #      for other data, it might be necessary to adapt this threshold.)
                    person = [ent for ent in doc.ents if ent.label_ == 'PERSON']
                    if len(person)/len(doc) >0.08:
                        continue

                    ### Filter out non-name mentions based on NER
                    all_vars = [name]
                    all_vars += variants[name]
    
                    # Find tuples of (offset, name) for all relevant name variants in the speech;
                    # use exact match (don't extract Vogelmann for Vogel etc.) 
                    results = []
                    offset_name_tuples = [] 
                    offset_name_tuples += [([(m.start(0), m.end(0)) for m in re.finditer(r'\b'+var+r'\b', speech)], var) for var in all_vars]
                    for offset_list, var in offset_name_tuples:
                        for start, end in offset_list:
                            results.append((var, start, end))

  
                    # First, filter out instances that are substrings of other hits, so that we
                    # don't count the same instance twice.
                    filtered = results.copy()
                    validated = []
                    for i in range(len(results)): 
                        if i == -1:
                            break
                        n1, s1, e1 = results[i]
                        for j in range(i+1, len(results)):
                            if i == j: 
                                continue
                            n2, s2, e2 = results[j]

                            # If one string is a substring of another name variant:
                            # check if both strings refer to the same text instance in the speech
                            if is_substring(n1, n2, s1, s2, e1, e2) == True:
                                # We want to keep the longer string:
                                # => remove n1 
                                if results[i] in filtered:
                                    filtered.remove(results[i]) 
                            elif is_substring(n2, n1, s2, s1, e2, e1) == True:
                                # => remove n2 
                                if results[j] in filtered:
                                    filtered.remove(results[j])  

                    # Now that we have the filtered list without duplicate entries,
                    # get the NER annotations and check whether the match is part of
                    # a larger named entity (e.g. "Vogel": Wolfgang Vogel => extracted  
                    # for Hans-Jochen Vogel)                     
                        
                    if len(filtered) >0:
                        validated = is_NER(name, doc, filtered, first_names)
                    # increase counts for this name/date
                    counts[eid][name][date]['count'] += sum(1 for item in validated)
                    # and store context of 100 characters to the left and right of the
                    # name mention for later inspection.
                    for item in validated:
                        start = item[1]-100
                        if start <0:
                            start = 0
                        end   = item[2]+100
                        if end >len(speech):
                            end = len(speech)
                        counts[eid][name][date]['mentions'].append((idx, speech[start:end]))

    outf  = open(result_file, 'w')
    contf = open(context_file, 'w')
    print("writing name counts to file ", result_file)
    outf.write("election_id\tname\tdate\tcount\n")
    print("writing name contexts to file ", context_file)
    contf.write("election_id\tname\tdate\tspeech_id\tcontext\n") 
    
    for eid in counts:
        for name in counts[eid]:
            for date in counts[eid][name]:
                outf.write(str(eid) + "\t" + name + "\t" + date + "\t" + str(counts[eid][name][date]['count']) + "\n")
                for idx, stext in counts[eid][name][date]['mentions']:
                    contf.write(str(eid) + "\t" + name + "\t" + date +"\t" + str(idx) + " " + stext + "\n")
    contf.close()
    outf.close()

    return


"""
Create a dictionary with name variants, using full names as keys
and last names and spelling variants of full/last name as values
(spelling variants can be extracted from the speeches in a pre-processing step, using the script <extract_name_variants.py>, and preferably cleaned up manually to remove noise).  

Return dictionary with name variants. 
Format:
{'Hans-Jochen Vogel': ['Vogel', 'Hans-Jochen Vogel', 'HansJochen Vogel'], ... }
"""
def read_name_variant_dic(file_path, name_dic):
	variants_dic = {}
	# start with full name and last name for each political actor
	for eid in name_dic:
		for full_name in name_dic[eid]:
			last_name = full_name.split(" ")[-1]

			if full_name not in variants_dic: 
				variants_dic[full_name] = [last_name]
			else:
				variants_dic[full_name].append(last_name)

	# now add additional name variants from file 
	with open(file_path) as json_file:
		spelling_variants_dic = json.load(json_file)
	for full_name in spelling_variants_dic:
		for name_var in spelling_variants_dic[full_name]:
			if name_var not in variants_dic[full_name]:
				variants_dic[full_name].append(name_var)
	return variants_dic 



"""
Now we iterate over all names (and their spelling variants) in the name_dic
    - select the speeches from csv files for the time periods we are interested in (3 months before/after election)
    - count the name mentions in the speeches 
	- count the country mentions in the speeches

####
### Format name_dic:    
{"260-1983-0306-L1": {
      "Helmut Kohl": {
         "country": "German Fed. Rep.",
         "date": [
            "198303",
            "198301",
            "198302",
            "198304",
            "198305",
            "198306"
         ]
      },
      "Hans-Jochen Vogel": {
         "country": "German Fed. Rep.",
         "date": [
            "198303",
            "198301",
            "198302",
            "198304",
            "198305",
            "198306"
         ]
      }
      ...
}
#####
### Format corpus/yyyymm.csv:  speech_id,speech


Return dictionary with counts for each eid and leader name.
"""

def count_name_mentions_in_corpus(name_dic, corpus_path,name_variants_file_path, first_names_list, result_file, context_file):
	dic = {}    
	variants_dic = read_name_variant_dic(name_variants_file_path, name_dic) 

	for eid in name_dic:
		for full_name in name_dic[eid]:
			dates = name_dic[eid][full_name]['date']
			country = name_dic[eid][full_name]['country'] 
			# Read all files included in dates (covering 3 months before/after election).
			for date in dates:
				csv_path = corpus_path + date + '.csv' 

				if path.exists(csv_path): 
					try:
						df = pd.read_csv(csv_path)
					except IOError as e:
						print("I/O error({0}): {1}".format(e.errno, e.strerror))
					except:
						print("Unexpected error:", sys.exc_info()[0], csv_path)    

	        	    # We first do a quick and dirty search for all name variants, based on 
		            # a string match (without any linguistic preprocessing/filtering).
		            # Then we only process those files where we found a string match for 
		            # a name/name variant,  validate the results and filter out false positives. 
					for name_variant in variants_dic[full_name]:
						hits = df[df['speech'].astype(str).str.contains(name_variant)]
						for idx, row in hits.iterrows():
							if eid not in dic:
								dic[eid] = {}
							if full_name not in dic[eid]:
								dic[eid][full_name] = {} 
							if date not in dic[eid][full_name]:
								dic[eid][full_name][date] = { 'speeches': [] }

							if (row[0], row.speech) not in dic[eid][full_name][date]['speeches']:
								dic[eid][full_name][date]['speeches'].append((row[0], row.speech))

    # We now have a dictionary with speeches that contain potential name mentions.
    # Next, we use spacy to filter out false positives before counting the mentions. 
	print("Filter and count...")
	filter_and_count(dic, variants_dic, first_names_list, result_file, context_file)

 
"""
Read a file with a list of first names (format: one name per line) and return as list.
"""
def read_first_names(first_name_file_path):
    first_names = []
    with open(first_name_file_path, 'r') as inf:
        for line in inf:
            first_names.append(line.strip())
    return first_names






if __name__ == "__main__": 

	name_file    = 'leader_names.csv'		# file with leader names and election dates
	country_file = 'countries_yes_no.json'	        # json dictionary with variants for each country 
	corpus_path  = 'corpus/'			# path to corpus directory with speeches

	name_variants_file_path = 'name_variants.json'
	first_names_file_path = 'first_names.txt'
	result_file = 'leader_counts.csv'		# Save counts to file.
	context_file = 'context.csv'			# Store the context for each leader mention
							# for validation.
	results_country_counts = 'country_counts.csv'

	count_name_mentions = True
	count_country_mentions = True
									
	print("... reading file with leader names:", name_file, " ...")
	start = time.time()
	name_dic, dates_of_interest = get_names_from_file(name_file)
	end = time.time()
	print("... done:\tfound", len(name_dic), " names ...\t(time: %0.2fs" % (end - start), ")")
	

	if count_name_mentions == True:	
		first_names_list = read_first_names(first_names_file_path)
		count_name_mentions_in_corpus(name_dic, corpus_path, name_variants_file_path, first_names_list, result_file, context_file)


	if count_country_mentions == True:
		print("... reading file with country names:", country_file, " ...")
		country_dic = read_country_dict(country_file)
		print("... counting countries ...")
		country_counts = count_country_mentions_in_speeches(name_dic, corpus_path, country_dic)
		print("... writing country counts to file ...")
		write_counts_to_file('country_counts.csv', country_counts)        
		print("... Done ...")


