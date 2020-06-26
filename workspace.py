import json as js
import time as tt
import string as stng
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import pandas as pd
import numpy as np
import re as re
import nltk as nltk

global nan
nan = np.nan

def runtime(start: float):
    '''
        (float) -> float
        computes runtime
        start: float value of start time, computed in main() function
    '''
    end_time = tt.time()
    print('RUNTIME: ', end_time-start)


def build_test_list(filename: str, test_range: int):
    '''
        (str) -> list
        helper function to build test list for smaller samples
        filename: str name of filename containing data
    '''
    lines = []
    with open(filename) as datastream:
        for i in range(test_range):
            lines.append(datastream.readline()) 
    return lines


def time_data_populate(filename: str, test_range: int, sourcename=False):
    time_pop_start = tt.time()
    output = open("workspace.txt","w", encoding="utf-8")    
    output_dict = {}
    lines = build_test_list(filename, test_range)

    for line in lines:
        js_obj = js.loads(line)
        date = (js_obj['published_at'].split())[0]
        time_published = (js_obj['published_at'].split())[1]
        domain = js_obj['source']['domain']
        article = js_obj['links']['permalink']

        if (sourcename != False) and (sourcename in domain):
            if date not in output_dict.keys():
                output_dict[date] = {}
            if domain not in output_dict[date].keys():
                output_dict[date][domain] = {}
            if article not in output_dict[date][domain].keys():
                output_dict[date][domain][article] = {}
            if 'total count' not in output_dict[date][domain].keys():
                output_dict[date][domain]['total count'] = {}
            output_dict[date][domain][article]['Time Published'] = time_published
        elif (sourcename == False):
            if date not in output_dict.keys():
                output_dict[date] = {}
            if domain not in output_dict[date].keys():
                output_dict[date][domain] = {}
            if article not in output_dict[date][domain].keys():
                output_dict[date][domain][article] = {}
            if 'total count' not in output_dict[date][domain].keys():
                output_dict[date][domain]['total count'] = {}
            output_dict[date][domain][article]['Time Published'] = time_published

    output.write(str(output_dict))
    output.close()
    print("\nTIME POPULATION", end = " ")
    runtime(time_pop_start)
    return output_dict


def freq_data_populate(filename: str, target_words: str, dict_input: dict, test_range: int, sourcename=False):
    '''
    '''
    freq_pop_start = tt.time()
    output = open("workspace.txt","w", encoding="utf-8")    
    lines = build_test_list(filename, test_range)
    targets = []

    with open(target_words) as datastream:
        for line in datastream.readlines():
            if line.strip()[-1] == '_':
                if ' ' in line.strip():
                    target = line.strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation)).split()
                    target[-1] += '_'
                    targets.append(list(target))
                else:
                    target = line.strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation))
                    target += '_'
                    targets.append(target)
            else:
                if ' ' in line.strip():
                    targets.append(tuple(line.strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation)).split()))
                else:
                    targets.append(line.strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation)))

    for line in lines:
        js_obj = js.loads(line)
        date = (js_obj['published_at'].split())[0]
        domain = js_obj['source']['domain']
        article = js_obj['links']['permalink']
        
        if (sourcename != False) and (sourcename not in domain):
            pass

        else:
            body_list = (js_obj['body']).lower()
            body_list = body_list.replace('-', ' ')
            body_list = body_list.translate(str.maketrans('','',stng.punctuation))
            body_list = body_list.split()

            for term in targets:
                if type(term) == str: ## edit for fatality/ies
                    if term[-1] == '_':
                        if term not in dict_input[date][domain]['total count'].keys():
                            term = term.translate(str.maketrans('','',stng.punctuation))
                            count = 0
                            for item in body_list:
                                if term in item:
                                    count += 1
                            term = (term + '_')
                            dict_input[date][domain]['total count'][term] = count
                        else:
                            term = term.translate(str.maketrans('','',stng.punctuation))
                            count = 0
                            for item in body_list:
                                if term in item:
                                    count += 1
                            term = (term + '_')
                            dict_input[date][domain]['total count'][term] += count
                        dict_input[date][domain][article][term] = count
                    else:
                        if term not in dict_input[date][domain]['total count'].keys():
                            dict_input[date][domain]['total count'][term] = body_list.count(term)
                        else:
                            dict_input[date][domain]['total count'][term] += body_list.count(term)
                        dict_input[date][domain][article][term] = body_list.count(term)
                
                elif type(term) == list:
                    term[-1] = term[-1].translate(str.maketrans('','',stng.punctuation))
                    token_count = 0
                    for i in range(len(body_list)):
                        if body_list[i] == term[0]:
                            count = 0
                            try:
                                for j in range(len(term)-1):
                                    if body_list[i+j] == term[j]:
                                        count += 1
                                if count == len(term)-1 and term[-1] in body_list[i+len(term)-1]:
                                    token_count += 1
                            except IndexError:
                                pass
                    term[-1] += '_'
                    if ' '.join(term) not in dict_input[date][domain]['total count'].keys():
                        dict_input[date][domain]['total count'][' '.join(term)] = token_count
                    else:
                        dict_input[date][domain]['total count'][' '.join(term)] += token_count
                    dict_input[date][domain][article][' '.join(term)] = token_count
                
                elif type(term) == tuple:
                    token_count = 0
                    for i in range(len(body_list)):
                        if body_list[i] == term[0]:
                            count = 0
                            try:
                                for j in range(len(term)):
                                    if body_list[i+j] == term[j]:
                                        count += 1
                                if count == len(term):
                                    token_count += 1
                            except IndexError:
                                pass
                    if ' '.join(term) not in dict_input[date][domain]['total count'].keys():
                        dict_input[date][domain]['total count'][' '.join(term)] = token_count
                    else:
                        dict_input[date][domain]['total count'][' '.join(term)] += token_count
                    dict_input[date][domain][article][' '.join(term)] = token_count

    output.write(str(dict_input))
    output.close()
    print("TARGET WORD POPULATION", end = " ")
    runtime(freq_pop_start)
    return dict_input


def main_process(dataset_filename: str, target_words_filename: str, test_range: int, sourcename=False):
    '''
        (str, str) -> dict
        Main processing function, takes string names of dataset and target word txt filenames
        Uses previous functions to populate output dictionary
    '''
    start_time = tt.time()
    print("\nSTART PROCESSING TEST AT: {} \nRUNNING...".format(tt.ctime()))

    test_dict = time_data_populate(dataset_filename, test_range, sourcename)
    output_dict = freq_data_populate(dataset_filename, target_words_filename, test_dict, test_range, sourcename)

    print('\n\nDONE')
    print("PROCESSING", end=" ")
    runtime(start_time)

    return output_dict

# main_test_process('aylien_data.jsonl', 'target_words.txt', 10)

############################## June 24th

# fmri = sns.load_dataset("fmri")
# print(fmri)

# df = pd.DataFrame(np.random.randn(4,1), index=['mon','wed','fri', 'sun'], columns=['sentiment'])
# df['date'] = df.index
# df['token'] = 'coronavirus'
# df2 = df.reindex(['mon','tue','wed','thu','fri','sat','sun'])
# df2['token'] = 'coronavirus'
# df2['date'] = ['2020-04','2020-03','2020-02','2020-01','2019-12','2019-11','2019-10']
# # print(df2['tue'])
# print(df2)

# paragraph = 'This is a test string! This is not an actual paragraph. Sentence about casualties? Sentence about death. Sentences and more sentences with fatalities and other words around it!'



# ax = sns.lineplot(x='date',y='sentiment',hue='token',err_style='bars',ci=69, data=df2)
# plt.show()

# import pandas as pd

# df = pd.read_csv('valence.csv')
# relevant = df[['Word','V.Mean.Sum']]

# words = df['Word'].tolist()
# ratings = df['V.Mean.Sum'].tolist()
# output = {}

# for i in range(len(words)):
#     output[words[i]] = ratings[i]

# with open('valence_ratings.txt', "w") as outfile:
#     outfile.write(str(output))

# import json as js
# import ast

# with open('test.txt', "r") as ratings:
#     contents = ratings.read()
#     valence_rating = eval(contents)

# print (type(valence_rating))


# df = pd.read_csv('valence.csv')
# relevant = df[['Word','V.Mean.Sum']]
# words = df['Word'].tolist()
# words[8289] = 'null'
# ratings = df['V.Mean.Sum'].tolist()
# valence_ratings = {}

# targets = ['coronavirus','dexamethasone']

# for i in range(len(words)):
#     valence_ratings[words[i]] = ratings[i]

# output_dict = {'word1':{'date1':[]}}

# with open('test.txt', "r") as in_file:
#     body = in_file.read()

# if len(re.findall('covid-19|covid 19', body, re.IGNORECASE)) > 0:
#     body = [nltk.tokenize.word_tokenize(sentence) for sentence in nltk.tokenize.sent_tokenize(body)]

# for i in range(len(body)):
    
#     valence = 0
#     count = 0
#     for word in body[i]:
#         if len(word) <= 2:
#             query = word
#             pass
#         elif word[-1] == 's' and len(word) > 2:
#             query = word + '?'
#         elif word[-1] != 's' and len(word) > 2:
#             query = word +'s?'
#         print(query)
#         for key in valence_ratings.keys():
#             if re.search(query, key):
#                 valence += valence_ratings[key]
#                 count += 1

#     if count > 0:
#         output_dict['word1']['date1'].append(valence/count)
#     else:
#         output_dict['word1']['date1'].append(NaN)
    
#     print(output_dict)

full_data = [['2020-04-05', '2020-04-04', '2020-04-03', '2020-04-02', '2020-04-01', '2020-03-24', '2020-03-21', '2020-03-20', '2020-03-20', '2020-03-20', '2020-03-17', '2020-03-15', '2020-03-14', '2020-03-12', '2020-03-10', '2020-03-08', '2020-03-04', '2020-02-21', '2020-02-16', '2020-02-14', '2020-02-11', '2020-02-11', '2020-02-11', '2020-02-10', '2020-02-10', '2020-02-10', '2020-01-31', '2019-11-01', '2019-11-02', '2019-11-03', '2019-11-04', '2019-11-05', '2019-11-06', '2019-11-07', '2019-11-08', '2019-11-09', '2019-11-10', '2019-11-11', '2019-11-12', '2019-11-13', '2019-11-14', '2019-11-15', '2019-11-16', '2019-11-17', '2019-11-18', '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22', '2019-11-23', '2019-11-24', '2019-11-25', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', '2019-12-01', '2019-12-02', '2019-12-03', '2019-12-04', '2019-12-05', '2019-12-06', '2019-12-07', '2019-12-08', '2019-12-09', '2019-12-10', '2019-12-11', '2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15', '2019-12-16', '2019-12-17', '2019-12-18', '2019-12-19', '2019-12-20', '2019-12-21', '2019-12-22', '2019-12-23', '2019-12-24', '2019-12-25', '2019-12-26', '2019-12-27', '2019-12-28', '2019-12-29', '2019-12-30', '2019-12-31', '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16', '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30', '2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07', '2020-02-08', '2020-02-09', '2020-02-12', '2020-02-13', '2020-02-15', '2020-02-17', '2020-02-18', '2020-02-19', '2020-02-20', '2020-02-22', '2020-02-23', '2020-02-24', '2020-02-25', '2020-02-26', '2020-02-27', '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-09', '2020-03-11', '2020-03-13', '2020-03-16', '2020-03-18', '2020-03-19', '2020-03-22', '2020-03-23', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31', '2020-04-05', '2020-04-05', '2020-04-05', '2020-04-05', '2020-04-05', '2020-04-05', '2020-04-04', '2020-04-03', '2020-04-02', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-03-24', '2020-03-21', '2020-03-21', '2020-03-21', '2020-03-20', '2020-03-20', '2020-03-20', '2020-03-20', '2020-03-20', '2020-03-20', '2020-03-20', '2020-03-17', '2020-03-17', '2020-03-17', '2020-03-17', '2020-03-17', '2020-03-15', '2020-03-15', '2020-03-14', '2020-03-12', '2020-03-10', '2020-03-08', '2020-03-08', '2020-03-08', '2020-03-08', '2020-03-04', '2020-02-21', '2020-02-16', '2020-02-16', '2020-02-14', '2020-02-11', '2020-02-10', '2020-01-31', '2019-11-01', '2019-11-02', '2019-11-03', '2019-11-04', '2019-11-05', '2019-11-06', '2019-11-07', '2019-11-08', '2019-11-09', '2019-11-10', '2019-11-11', '2019-11-12', '2019-11-13', '2019-11-14', '2019-11-15', '2019-11-16', '2019-11-17', '2019-11-18', '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22', '2019-11-23', '2019-11-24', '2019-11-25', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', '2019-12-01', '2019-12-02', '2019-12-03', '2019-12-04', '2019-12-05', '2019-12-06', '2019-12-07', '2019-12-08', '2019-12-09', '2019-12-10', '2019-12-11', '2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15', '2019-12-16', '2019-12-17', '2019-12-18', '2019-12-19', '2019-12-20', '2019-12-21', '2019-12-22', '2019-12-23', '2019-12-24', '2019-12-25', '2019-12-26', '2019-12-27', '2019-12-28', '2019-12-29', '2019-12-30', '2019-12-31', '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16', '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30', '2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07', '2020-02-08', '2020-02-09', '2020-02-12', '2020-02-13', '2020-02-15', '2020-02-17', '2020-02-18', '2020-02-19', '2020-02-20', '2020-02-22', '2020-02-23', '2020-02-24', '2020-02-25', '2020-02-26', '2020-02-27', '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-09', '2020-03-11', '2020-03-13', '2020-03-16', '2020-03-18', '2020-03-19', '2020-03-22', '2020-03-23', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31', '2020-04-05', '2020-04-04', '2020-04-03', '2020-04-02', '2020-04-01', '2020-03-24', '2020-03-21', '2020-03-20', '2020-03-17', '2020-03-15', '2020-03-14', '2020-03-12', '2020-03-10', '2020-03-08', '2020-03-04', '2020-02-21', '2020-02-16', '2020-02-14', '2020-02-11', '2020-02-10', '2020-01-31', '2019-11-01', '2019-11-02', '2019-11-03', '2019-11-04', '2019-11-05', '2019-11-06', '2019-11-07', '2019-11-08', '2019-11-09', '2019-11-10', '2019-11-11', '2019-11-12', '2019-11-13', '2019-11-14', '2019-11-15', '2019-11-16', '2019-11-17', '2019-11-18', '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22', '2019-11-23', '2019-11-24', '2019-11-25', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', '2019-12-01', '2019-12-02', '2019-12-03', '2019-12-04', '2019-12-05', '2019-12-06', '2019-12-07', '2019-12-08', '2019-12-09', '2019-12-10', '2019-12-11', '2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15', '2019-12-16', '2019-12-17', '2019-12-18', '2019-12-19', '2019-12-20', '2019-12-21', '2019-12-22', '2019-12-23', '2019-12-24', '2019-12-25', '2019-12-26', '2019-12-27', '2019-12-28', '2019-12-29', '2019-12-30', '2019-12-31', '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16', '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30', '2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07', '2020-02-08', '2020-02-09', '2020-02-12', '2020-02-13', '2020-02-15', '2020-02-17', '2020-02-18', '2020-02-19', '2020-02-20', '2020-02-22', '2020-02-23', '2020-02-24', '2020-02-25', '2020-02-26', '2020-02-27', '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-09', '2020-03-11', '2020-03-13', '2020-03-16', '2020-03-18', '2020-03-19', '2020-03-22', '2020-03-23', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31', '2020-04-05', '2020-04-05', '2020-04-05', '2020-04-05', '2020-04-05', '2020-04-04', '2020-04-03', '2020-04-02', '2020-04-01', '2020-04-01', '2020-04-01', '2020-04-01', '2020-03-24', '2020-03-21', '2020-03-20', '2020-03-20', '2020-03-17', '2020-03-15', '2020-03-14', '2020-03-12', '2020-03-10', '2020-03-08', '2020-03-04', '2020-02-21', '2020-02-16', '2020-02-14', '2020-02-11', '2020-02-10', '2020-01-31', '2019-11-01', '2019-11-02', '2019-11-03', '2019-11-04', '2019-11-05', '2019-11-06', '2019-11-07', '2019-11-08', '2019-11-09', '2019-11-10', '2019-11-11', '2019-11-12', '2019-11-13', '2019-11-14', '2019-11-15', '2019-11-16', '2019-11-17', '2019-11-18', '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22', '2019-11-23', '2019-11-24', '2019-11-25', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', '2019-12-01', '2019-12-02', '2019-12-03', '2019-12-04', '2019-12-05', '2019-12-06', '2019-12-07', '2019-12-08', '2019-12-09', '2019-12-10', '2019-12-11', '2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15', '2019-12-16', '2019-12-17', '2019-12-18', '2019-12-19', '2019-12-20', '2019-12-21', '2019-12-22', '2019-12-23', '2019-12-24', '2019-12-25', '2019-12-26', '2019-12-27', '2019-12-28', '2019-12-29', '2019-12-30', '2019-12-31', '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16', '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30', '2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07', '2020-02-08', '2020-02-09', '2020-02-12', '2020-02-13', '2020-02-15', '2020-02-17', '2020-02-18', '2020-02-19', '2020-02-20', '2020-02-22', '2020-02-23', '2020-02-24', '2020-02-25', '2020-02-26', '2020-02-27', '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-09', '2020-03-11', '2020-03-13', '2020-03-16', '2020-03-18', '2020-03-19', '2020-03-22', '2020-03-23', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31', '2020-04-05', '2020-04-04', '2020-04-03', '2020-04-02', '2020-04-01', '2020-03-24', '2020-03-21', '2020-03-20', '2020-03-17', '2020-03-15', '2020-03-14', '2020-03-12', '2020-03-10', '2020-03-08', '2020-03-04', '2020-02-21', '2020-02-16', '2020-02-14', '2020-02-11', '2020-02-10', '2020-01-31', '2019-11-01', '2019-11-02', '2019-11-03', '2019-11-04', '2019-11-05', '2019-11-06', '2019-11-07', '2019-11-08', '2019-11-09', '2019-11-10', '2019-11-11', '2019-11-12', '2019-11-13', '2019-11-14', '2019-11-15', '2019-11-16', '2019-11-17', '2019-11-18', '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22', '2019-11-23', '2019-11-24', '2019-11-25', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', '2019-12-01', '2019-12-02', '2019-12-03', '2019-12-04', '2019-12-05', '2019-12-06', '2019-12-07', '2019-12-08', '2019-12-09', '2019-12-10', '2019-12-11', '2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15', '2019-12-16', '2019-12-17', '2019-12-18', '2019-12-19', '2019-12-20', '2019-12-21', '2019-12-22', '2019-12-23', '2019-12-24', '2019-12-25', '2019-12-26', '2019-12-27', '2019-12-28', '2019-12-29', '2019-12-30', '2019-12-31', '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16', '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30', '2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07', '2020-02-08', '2020-02-09', '2020-02-12', '2020-02-13', '2020-02-15', '2020-02-17', '2020-02-18', '2020-02-19', '2020-02-20', '2020-02-22', '2020-02-23', '2020-02-24', '2020-02-25', '2020-02-26', '2020-02-27', '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-09', '2020-03-11', '2020-03-13', '2020-03-16', '2020-03-18', '2020-03-19', '2020-03-22', '2020-03-23', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31'], ['coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'coronavirus', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'covid-19|covid 19', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'epidemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'pandemic', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout', 'breakout'], [nan, nan, nan, nan, nan, nan, nan, 5.346, 5.017, 5.047, 5.045, nan, 5.033, nan, 4.796, nan, nan, 5.156, nan, 5.216, 5.156, 5.489, 5.827, 5.088, 5.301, 5.13, 5.223, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 5.162, 5.387, 5.081, 5.369, 5.37, 5.338, 5.375, 5.252, 5.088, 5.091, 5.029, 5.016, 5.314, 5.556, 5.54, 5.188, 5.322, 5.249, 5.118, 5.095, 5.285, 5.306, 5.231, 5.072, 5.346, 5.017, 5.304, 5.257, 5.037, 5.047, 5.165, 5.14, 5.045, 5.051, 5.176, 5.036, 5.167, 5.227, 5.033, 5.22, 4.796, 5.404, 5.437, 5.362, 5.373, 5.201, 5.156, 5.276, 5.226, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 5.162, 5.588, 5.387, 5.081, 5.338, 5.375, 5.252, 5.088, 5.091, 5.029, 5.322, nan, 5.285, 5.072, 5.337, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]]

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

daterange = pd.date_range(start='2019-11-01', end='2020-04-05')
alldates = []

for date in daterange:
    alldates.append(str(date).split()[0])

dates = full_data[0]
tokens = full_data[1]
values = full_data[2]

sentiment_data = {'Dates': dates, 'Tokens': tokens, 'Ratings': values}

fig, ax = plt.subplots()
sb.set(style="whitegrid")

data_in_sentiment = pd.DataFrame(data=sentiment_data)
data_in_sentiment.sort_values(by=['Dates'], ascending=True, inplace=True)
sentiment = sb.pointplot(x='Dates', y='Ratings', hue='Tokens', dashes=False, palette="tab10", linewidth=2.0, data=data_in_sentiment)
sentiment.set(xlabel='Date', ylabel='Mean Sentiment Rating')
plt.title('Sentiment Plot Cluster1 canada.ca')

plt.xticks(rotation=80)
# for label in sentiment.xaxis.get_ticklabels()[::2]:
#     label.set_visible(False)
n = 7
[l.set_visible(False) for (i,l) in enumerate(sentiment.xaxis.get_ticklabels()) if i % n != 0]

plt.show()
