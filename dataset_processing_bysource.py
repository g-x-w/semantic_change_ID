import json as js
import time as tt
import re as re
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import numpy as np
import seaborn as sb
import pandas as pd
import csv as csv
import nltk as nltk
import string as string
import ast as ast

def runtime(start):
    '''
        (float) -> float

        computes runtime

        start: float value of start time, computed in main() function
    '''
    end_time = tt.time()
    return (end_time-start)


def populate_data(input_data_filename: str, target_words_filename: str, sourcename: str):
    '''
        (str, str, str) -> [[str],{{{{str: int}}}}

        Takes JSON lines file and returns list with queried targets 
        and quadruple-nested dictionary of processed data

        Input: filenames of input dataset and target words .txt file and sourcename to query
        Output:
        [
            [target1, target2, ..., targetn],
            {date1: {
                source1: {
                    total count: {word1: count, word2: count, ..., wordn: count},
                    article1: {time: '16:00:00', word1: count, word2: count, ..., wordn: count},
                    article2: {time: '16:00:00', word1: count, word2: count, ..., wordn: count},
                    ...
                    articlen: {time: '16:00:00', word1: count, word2: count, ..., wordn: count}
                source2: {...},
                ...
                sourcen: {...}
                }
            date2: {
                source1: {...}, 
                source2: {...},
                ...
                sourcen: {...}
                }
            ...
            daten: {
                source1: {...}, 
                source2: {...},
                ...
                sourcen: {...}
                }    
            }
        ]
    '''

    output = open("output_data_full.txt", "w", encoding="utf-8")
    output_dict = {}
    targets = []

    with open(target_words_filename) as targetstream:
        for line in targetstream.readlines():
            targets.append(line.strip())

    with open(input_data_filename) as datastream:
        for line in datastream:
            js_obj = js.loads(line)
            domain = js_obj['source']['domain']

            if sourcename == domain:        # == vs in breaks script
                date = (js_obj['published_at'].split())[0]
                time_published = (js_obj['published_at'].split())[1]
                article = js_obj['links']['permalink']
                body = js_obj['body']

                if date not in output_dict.keys():
                    output_dict[date] = {}
                if domain not in output_dict[date].keys():
                    output_dict[date][domain] = {}
                if article not in output_dict[date][domain].keys():
                    output_dict[date][domain][article] = {}
                if 'total count' not in output_dict[date][domain].keys():
                    output_dict[date][domain]['total count'] = {}
                output_dict[date][domain][article]['Time Published'] = time_published

                for term in targets:
                    if term not in output_dict[date][domain]['total count'].keys():
                        output_dict[date][domain]['total count'][term] = len(re.findall(term, body, re.IGNORECASE))
                    else:
                        output_dict[date][domain]['total count'][term] += len(re.findall(term, body, re.IGNORECASE))
                    output_dict[date][domain][article][term] = len(re.findall(term, body, re.IGNORECASE))

    processed_data = [targets, output_dict]
    output.write(str(processed_data))
    output.close()
    return processed_data


def dataframe_setup(processed_input_data: list, sourcename: str):
    '''
        (str, str) -> [[str],[str], [{str:[int]}]]
        
        Pulls relevant frequency traces from processed dataset output by populate_data
        and cleans it up to prepare for dataframe formatting for graph generation

        Input: processed dataset returned from dataset_processing.py
        Output:
        [
            [target1, target2, target3, ..., targetn], 
            [date1, date2, date3, ... , daten],
            [
                {source1: [word1day1_ct, word2day1_ct, ... , wordnday1_ct], 
                source2: [word1day1_ct, word2day1_ct, ... , wordnday1_ct]},
                {source1: [word1day2_ct, word2day2_ct, ... , wordnday2_ct], 
                source2: [word1day2_ct, word2day2_ct, ... , wordnday2_ct]}
            ]
        ]
    '''
    processed_data = processed_input_data[1]
    targets = processed_input_data[0]
    date_list = []
    source_count_list = []
    file_out_name = '{}_dataframe_output.txt'.format(sourcename)
    outfile = open(file_out_name, "w", encoding="utf-8")

    for date in processed_data.keys():
        date_list.append(date)
        source_counts = {}

        for target in targets:
            num_sources = 0
            for source in processed_data[date].keys():
                if source not in source_counts.keys():
                    source_counts[source] = [[],[]]
                source_counts[source][0].append(processed_data[date][source]['total count'][target])
                for article in processed_data[date][source].keys():
                    if article == 'total count':
                        pass
                    else:
                        if processed_data[date][source][article][target] > 0:
                            num_sources += 1
            source_counts[source][1].append(num_sources)
        source_count_list.append(source_counts)

    output = [targets, date_list, source_count_list]
    outfile.write(str(output))
    outfile.close()
    return output


def graphing_type(input_list: list, target_words_filename: str, sourcename: str):
    '''
        (list, str, str) -> png image files

        DESCRIPTION HERE

        Input: list from dataframe_setup in format [[targets], [dates], [{sources:[counts]}]]
        Output: None and graphs in .png format
    '''
    title = "{}_{}_FullTrace.png".format(sourcename, target_words_filename)
    tok_title = "Token Frequency for {} from {}".format(target_words_filename, sourcename)
    type_title = "Type Frequency for {} from {}".format(target_words_filename, sourcename)

    fig, ax = plt.subplots()
    sb.set(style="whitegrid")

    tok_count_list = []
    for i in range(len(input_list[2])):
        for key in input_list[2][i].keys():
            tok_count_list.append(input_list[2][i][key][0])

    type_count_list = []
    for i in range(len(input_list[2])):
        for key in input_list[2][i].keys():
            type_count_list.append(input_list[2][i][key][1])

    num_tokens = len(tok_count_list[0])
    earliest = min(input_list[1])
    latest = max(input_list[1])
    daterange = pd.date_range(start=earliest, end=latest)

    for date in daterange:
        if str(date).split()[0] not in input_list[1]:
            input_list[1].append(str(date).split()[0])
            tok_count_list.append([0]*num_tokens)
            type_count_list.append([0]*num_tokens)
        else:
            pass

    plt.subplot(211)
    data_in_tok = pd.DataFrame(data=tok_count_list, index=input_list[1], columns=input_list[0])
    token_freq = sb.lineplot(data=data_in_tok, dashes=False, palette="tab10", linewidth=2.0)
    token_freq.set(xlabel='Date', ylabel='Occurrences')
    token_freq.xaxis.set_major_locator(tk.MultipleLocator(7))
    plt.yscale('log')
    plt.title(tok_title)
    plt.xticks(rotation=40)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0.)

    plt.subplot(212)
    data_in_type = pd.DataFrame(data=type_count_list, index=input_list[1], columns=input_list[0])
    type_freq = sb.lineplot(data=data_in_type, dashes=False, palette="tab10", linewidth=2.0)
    type_freq.set(xlabel='Date', ylabel='Number of Articles with Occurrence(s)')
    type_freq.xaxis.set_major_locator(tk.MultipleLocator(7))
    plt.yscale('log')
    plt.title(type_title)
    plt.xticks(rotation=40)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0.)

    print(data_in_type)
    print(data_in_tok)

    graph = plt.gcf()
    graph.set_size_inches((11, 8.5), forward=False)
    graph.tight_layout()
    graph.savefig(title, dpi=600)
    # plt.show()

    return None


def csv_output(input_list: list, sourcename: str):
    '''
        (list, str) -> csv

        DESCRIPTION HERE

        Input: list from dataframe_setup in format [[targets], [dates], [{sources:[counts]}]]
        Output: None and human-readable trace data in .csv format
    '''
    outfile = "{}_type_tok_freq.csv".format(sourcename)

    row_1 = ['Date']
    rows = []

    for i in range(len(input_list[1])):
        rows.append([])

    for i in range(len(input_list[0])):
        row_1.append(input_list[0][i])

    for i in range(len(input_list[1])):
        for j in range(len(input_list[0])):
            rows[i].append(input_list[2][i][sourcename][0][j])
    
    for i in range(len(rows)):
        rows[i].insert(0, input_list[1][i])

    with open(outfile, "w", newline='') as outfile:
        wr = csv.writer(outfile)
        wr.writerow(row_1)
        for line in rows:
            wr.writerow(line)

    return None


def get_sentiment_ratings():
    # df = pd.read_csv('valence.csv')
    # relevant = df[['Word','V.Mean.Sum']]
    # words = df['Word'].tolist()
    # words[8289] = 'null'
    # ratings = df['V.Mean.Sum'].tolist()
    # valence_ratings = {}

    # for i in range(len(words)):
    #     valence_ratings[words[i]] = ratings[i]
    
    # return(valence_ratings)

    with open('valence_ratings_adjusted.txt', "r") as in_file:
        valence_ratings = ast.literal_eval(in_file.read())

    return (valence_ratings)


def get_pos_tag(word_input):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ, "N": nltk.corpus.wordnet.NOUN, "V": nltk.corpus.wordnet.VERB, "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


def populate_data_sentiment(input_data_filename: str, target_words_filename: str, sourcename: str, valence_ratings: dict):
    '''
        (str, str, str) -> [[str],{{{{str: int}}}}

        Takes JSON lines file and returns list with queried targets 
        and quadruple-nested dictionary of processed data

        Input: filenames of input dataset and target words .txt file and sourcename to query
        Output:
        [
            [target1, target2, target3],
            {word1: {date1: [val, val, val, val], date2: [val, val, val, val]}, 
            word2: {date1: [val, val, val, val], date2: [val, val, val, val]}, 
            ...} 
        ]
    '''

    output = open("output_data_full.txt", "w", encoding="utf-8")
    output_dict = {}
    targets = []

    stopwords = set(nltk.corpus.stopwords.words('english'))

    with open(target_words_filename) as targetstream:
        for line in targetstream.readlines():
            targets.append(line.strip())
    
    for target in targets:
        output_dict[target] = {}
    
    article_count = 0

    with open(input_data_filename) as datastream:
        for line in datastream:
            js_obj = js.loads(line)
            domain = js_obj['source']['domain']

            if sourcename == domain:       
                date = (js_obj['published_at'].split())[0]
                body = js_obj['body']
                for target in targets:
                    if date not in output_dict[target].keys():
                        output_dict[target][date] = []

                    if re.search(target, body, re.IGNORECASE):
                        article_count += 1
                        print('Target found in article:', article_count, 'Target:', target)
                        body2 = nltk.tokenize.sent_tokenize(body)

                        for i in range(len(body2)):     # for each sentence in the body
                            if re.search(target, body2[i], re.IGNORECASE):
                                print('Target found in sentence')
                                cleaned = []
                                body2[i] = nltk.tokenize.word_tokenize(body2[i])

                                for token in body2[i]:
                                    if token.lower() not in stopwords and token not in string.punctuation:
                                        cleaned.append(token)

                                valence = 0
                                # count = 0
                                for word in cleaned:    # for each word in the processed sentence
                                    if len(word) <= 2:
                                        query = word
                                    elif word[-1] == 's' and len(word) > 2:
                                        query = word + '?'
                                    elif word[-1] != 's' and len(word) > 2:
                                        query = word +'s?'

                                    for key in valence_ratings.keys():
                                        if re.search(query, key, re.IGNORECASE) and abs(valence_ratings[key]) > abs(valence):
                                            print('Valence updated')
                                            valence = valence_ratings[key]
                                            # valence += valence_ratings[key]
                                            # count += 1
                                
                                if valence != 0:
                                    output_dict[target][date].append(valence)

                                # if count > 0:
                                #     output_dict[target][date].append(round(valence/count, 3))
                    else: 
                        output_dict[target][date].append(np.nan)
    
    processed_data = [targets, output_dict]
    output.write(str(processed_data))
    output.close()
    return processed_data


def dataframe_setup_sentiment(processed_input_data: list, sourcename: str):
    '''
        (str, str) -> [[str],[str], [{str:[int]}]]
        
        Pulls relevant frequency traces from processed dataset output by populate_data
        and cleans it up to prepare for dataframe formatting for graph generation

        Input: processed dataset returned from dataset_processing.py
        Output:
        [
            [target1, target2, target3, ..., targetn], 
            [date1, date2, date3, ... , daten],
            [
                {source1: [word1day1_ct, word2day1_ct, ... , wordnday1_ct], 
                source2: [word1day1_ct, word2day1_ct, ... , wordnday1_ct]},
                {source1: [word1day2_ct, word2day2_ct, ... , wordnday2_ct], 
                source2: [word1day2_ct, word2day2_ct, ... , wordnday2_ct]}
            ]
        ]
    '''
    processed_data = processed_input_data[1]
    targets = processed_input_data[0]
    date_list = []
    value_list = []
    token_list = []

    min_parse = []
    max_parse = []
    for key in processed_input_data[1].keys():
        min_parse.append(min(processed_input_data[1][key].keys()))
        max_parse.append(max(processed_input_data[1][key].keys()))
    mindate = min(min_parse)
    maxdate = max(max_parse)

    daterange = pd.date_range(start=mindate, end=maxdate)
    file_out_name = '{}_dataframe_output.txt'.format(sourcename)
    outfile = open(file_out_name, "w", encoding="utf-8")

    for target in targets:
        for date in daterange:
            if str(date).split()[0] not in processed_data[target].keys():
                processed_data[target][str(date).split()[0]] = [np.nan]

    for target in targets:
        for date in processed_data[target].keys():
            if np.isnan(processed_data[target][date][0]):
                value_list.append(np.nan)
                date_list.append(date)
                token_list.append(target)

            elif not np.isnan(processed_data[target][date][0]):
                for i in range(len(processed_data[target][date])):
                    value_list.append(processed_data[target][date][i])
                    date_list.append(date)
                    token_list.append(target)

    output = {'Dates': date_list, 'Tokens': token_list, 'Ratings': value_list}
    outfile.write(str(output))
    outfile.close()
    return output


def graphing_sentiment(input_list: list, sentiment_input: list, target_words_filename: str, sourcename: str):
    '''
        (list, str, str) -> png image files

        DESCRIPTION HERE

        Input: list from dataframe_setup in format [[targets], [dates], [{sources:[counts]}]]
        Output: None and graphs in .png format
    '''
    fig, ax = plt.subplots()
    sb.set(style="whitegrid")

    title = "{}_{}_FullTrace.png".format(sourcename, target_words_filename)
    tok_title = "Token Frequency for {} from {}".format(target_words_filename, sourcename)
    sentiment_title = "Sentiment Plot for {} from {}".format(target_words_filename, sourcename)

    tok_count_list = []
    for i in range(len(input_list[2])):
        for key in input_list[2][i].keys():
            tok_count_list.append(input_list[2][i][key][0])

    num_tokens = len(tok_count_list[0])
    earliest = min(input_list[1])
    latest = max(input_list[1])
    daterange = pd.date_range(start=earliest, end=latest)

    for date in daterange:
        if str(date).split()[0] not in input_list[1]:
            input_list[1].append(str(date).split()[0])
            tok_count_list.append([0]*num_tokens)
        else:
            pass

    plt.subplot(211)
    data_in_tok = pd.DataFrame(data=tok_count_list, index=input_list[1], columns=input_list[0])
    token_freq = sb.lineplot(data=data_in_tok, dashes=False, palette="tab10", linewidth=2.0)
    token_freq.set(xlabel='Date', ylabel='Occurrences')
    token_freq.xaxis.set_major_locator(tk.MultipleLocator(7))
    plt.yscale('log')
    plt.title(tok_title)
    plt.xticks(rotation=40)

    plt.subplot(212)
    data_in_sentiment = pd.DataFrame(data=sentiment_input)
    data_in_sentiment.sort_values(by=['Dates'], ascending=True, inplace=True)
    sentiment = sb.pointplot(x='Dates', y='Ratings', hue='Tokens', dashes=False, palette="tab10", linewidth=2.0, data=data_in_sentiment, ci=None)
    sentiment.set(xlabel='Date', ylabel='Mean Sentiment Rating')
    plt.title(sentiment_title)

    plt.xticks(rotation=80)
    n = 7
    [l.set_visible(False) for (i,l) in enumerate(sentiment.xaxis.get_ticklabels()) if i % n != 0]

    # graph = plt.gcf()
    # graph.set_size_inches((11, 8.5), forward=False)
    # graph.tight_layout()
    # graph.savefig(title, dpi=600)

    plt.show()

    return None


def main(input_data_filename: str, target_words_filename: str, sourcename: str, trace: int): # 1 = sentiment
    # print("\n\tTOTAL TRACE START: {}".format(tt.ctime()))     # Commented when running from main.py to clean up terminal output 
    main_start = tt.time()

    if trace == 0:
        print("\n\t\t{:25} {}".format('TIME POPULATION START:', tt.ctime()))
        time_pop_start = tt.time()
        processed_data = populate_data(input_data_filename, target_words_filename, sourcename)
        print("\t\t{:25} {:.2f}s".format('TIME POPULATION RUNTIME:', runtime(time_pop_start)))

        print("\n\t\t{:25} {}".format('DATAFRAME SETUP START:', tt.ctime()))
        dataframe_start = tt.time()
        dataframe = dataframe_setup(processed_data, sourcename)
        print("\t\t{:25} {:.2f}μs".format('DATAFRAME SETUP RUNTIME:', 1000000*runtime(dataframe_start)))
        
        print("\n\t\t{:25} {}".format('CSV OUTPUT START:', tt.ctime()))
        csv_start = tt.time()
        csv_output(dataframe, sourcename)
        print("\t\t{:25} {:.2f}μs".format('CSV OUTPUT RUNTIME:', 1000000*runtime(csv_start)))

        print("\n\t\tGRAPHING START: ", tt.ctime())
        graph_start = tt.time()
        graphing_type(dataframe, target_words_filename, sourcename)
        print("\t\tGRAPHING RUNTIME: {:.2f}".format(runtime(graph_start)))

        print("\n\t\tTOTAL TRACE RUNTIME: {:.2f}".format(runtime(main_start)))
    
    elif trace == 1:
        # print("\n\t\t{:25} {}".format('TIME POPULATION START:', tt.ctime()))
        # time_pop_start = tt.time()
        # processed_data = populate_data(input_data_filename, target_words_filename, sourcename)
        # print("\t\t{:25} {:.2f}s".format('TIME POPULATION RUNTIME:', runtime(time_pop_start)))

        # print("\n\t\t{:25} {}".format('DATAFRAME SETUP START:', tt.ctime()))
        # dataframe_start = tt.time()
        # dataframe = dataframe_setup(processed_data, sourcename)
        # print("\t\t{:25} {:.2f}μs".format('DATAFRAME SETUP RUNTIME:', 1000000*runtime(dataframe_start)))
        
        # print("\n\t\t{:25} {}".format('CSV OUTPUT START:', tt.ctime()))
        # csv_start = tt.time()
        # csv_output(dataframe, sourcename)
        # print("\t\t{:25} {:.2f}μs".format('CSV OUTPUT RUNTIME:', 1000000*runtime(csv_start)))

        print("\n\t\t{:25} {}".format('KUPERMAN TRACE START:', tt.ctime()))
        sentiment_start = tt.time()
        valences = get_sentiment_ratings()
        print("\t\t{:25} {:.2f}s".format('KUPERMAN TRACE RUNTIME:', runtime(sentiment_start)))

        print("\n\t\t{:25} {}".format('SENTIMENT DATA START:', tt.ctime()))
        sent_pop_start = tt.time()
        processed_sent_data = populate_data_sentiment(input_data_filename, target_words_filename, sourcename, valences)
        print("\t\t{:25} {:.2f}s".format('SENTIMENT DATA RUNTIME:', runtime(sent_pop_start)))

        # print("\n\t\t{:25} {}".format('SENTI FRAME START:', tt.ctime()))
        # sent_frame_start = tt.time()
        # sent_frame = dataframe_setup_sentiment(processed_sent_data, sourcename)
        # print("\t\t{:25} {:.2f}s".format('SENTI FRAME RUNTIME:', runtime(sent_frame_start)))

        # print("\n\t\tGRAPHING START: ", tt.ctime())
        # graph_start = tt.time()
        # graphing_sentiment(dataframe, sent_frame, target_words_filename, sourcename)
        # print("\t\tGRAPHING RUNTIME: {:.2f}".format(runtime(graph_start)))

        # print("\n\t\tTOTAL TRACE RUNTIME: {:.2f}".format(runtime(main_start)))

    return None

main('aylien_data.jsonl', 'cluster1_coronavirus.txt', 'cnn.com', 1)

##

# nan = np.nan

# temp = [['coronavirus', 'covid-19|covid 19', 'epidemic', 'pandemic', 'breakout'], {'coronavirus': {'2020-04-05': [nan], '2020-04-04': [nan], '2020-04-03': [nan], '2020-04-02': [nan], '2020-04-01': [nan, nan, nan], '2020-03-24': [nan], '2020-03-21': [nan], '2020-03-20': [-3.38, -3.38, -3.8], '2020-03-17': [-3.48], '2020-03-15': [nan], '2020-03-14': [-3.48], '2020-03-12': [nan], '2020-03-10': [-3.38], '2020-03-08': [nan], '2020-03-04': [nan], '2020-02-21': [-1.73], '2020-02-16': [nan], '2020-02-14': [-3.45], '2020-02-11': [-1.73, -2.82, 2.7], '2020-02-10': [-3.66, 2.05, -3.31], '2020-01-31': [-3.45]}, 'covid-19|covid 19': {'2020-04-05': [-3.45, 2.49, -3.53, -2.56, -2.71, -3.53], '2020-04-04': [-3.35], '2020-04-03': [-3.45], '2020-04-02': [-3.48], '2020-04-01': [-3.66, -3.66, -3.01, -3.45, 2.89, -3.04, -3.45, -2.87, -3.45, -3.45, -3.01], '2020-03-24': [-3.45], '2020-03-21': [-3.45, 2.83, -3.53], '2020-03-20': [-3.38, -3.38, -3.06, -3.45, -3.53, -3.8, -3.45], '2020-03-17': [-3.45, -3.48, -3.66, -3.16, -3.48], '2020-03-15': [-3.45, -3.45], '2020-03-14': [-3.48], '2020-03-12': [-3.45], '2020-03-10': [-3.38], '2020-03-08': [-3.45, 2.05, -3.45, -3.04], '2020-03-04': [-3.45], '2020-02-21': [-1.73], '2020-02-16': [2.7, -2.17], '2020-02-14': [nan], '2020-02-11': [nan], '2020-02-10': [nan], '2020-01-31': [nan]}, 'epidemic': {'2020-04-05': [nan], '2020-04-04': [nan], '2020-04-03': [nan], '2020-04-02': [nan], '2020-04-01': [nan, nan, nan], '2020-03-24': [nan], '2020-03-21': [nan], '2020-03-20': [nan, nan], '2020-03-17': [nan], '2020-03-15': [nan], '2020-03-14': [nan], '2020-03-12': [nan], '2020-03-10': [nan], '2020-03-08': [nan], '2020-03-04': [nan], '2020-02-21': [nan], '2020-02-16': [nan], '2020-02-14': [nan], '2020-02-11': [nan], '2020-02-10': [nan], '2020-01-31': [nan]}, 'pandemic': {'2020-04-05': [-3.45, 2.04, 2.49, -3.53, -3.53], '2020-04-04': [-3.35], '2020-04-03': [-3.45], '2020-04-02': [-3.48], '2020-04-01': [-3.66, -3.66, -2.87, nan], '2020-03-24': [-3.45], '2020-03-21': [-3.53], '2020-03-20': [-3.35, nan], '2020-03-17': [nan], '2020-03-15': [nan], '2020-03-14': [nan], '2020-03-12': [nan], '2020-03-10': [nan], '2020-03-08': [nan], '2020-03-04': [nan], '2020-02-21': [nan], '2020-02-16': [nan], '2020-02-14': [nan], '2020-02-11': [nan], '2020-02-10': [nan], '2020-01-31': [nan]}, 'breakout': {'2020-04-05': [nan], '2020-04-04': [nan], '2020-04-03': [nan], '2020-04-02': [nan], '2020-04-01': [nan, nan, nan], '2020-03-24': [nan], '2020-03-21': [nan], '2020-03-20': [nan, nan], '2020-03-17': [nan], '2020-03-15': [nan], '2020-03-14': [nan], '2020-03-12': [nan], '2020-03-10': [nan], '2020-03-08': [nan], '2020-03-04': [nan], '2020-02-21': [nan], '2020-02-16': [nan], '2020-02-14': [nan], '2020-02-11': [nan], '2020-02-10': [nan], '2020-01-31': [nan]}}]

# sentiment_input = dataframe_setup_sentiment(temp, 'canada.ca')

# fig, ax = plt.subplots()
# sb.set(style="whitegrid")

# plt.subplot(212)
# data_in_sentiment = pd.DataFrame(data=sentiment_input)
# data_in_sentiment.sort_values(by=['Dates'], ascending=True, inplace=True)
# sentiment = sb.pointplot(x='Dates', y='Ratings', hue='Tokens', dashes=False, palette="tab10", linewidth=2.0, data=data_in_sentiment, ci=None)
# sentiment.set(xlabel='Date', ylabel='Mean Sentiment Rating')

# plt.xticks(rotation=80)
# n = 7
# [l.set_visible(False) for (i,l) in enumerate(sentiment.xaxis.get_ticklabels()) if i % n != 0]


# plt.show()
