import json as js
import time as tt
import re as re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import csv as csv

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

            if sourcename in domain:
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

    output.write(str(output_dict))
    output.close()
    processed_data = [targets, output_dict]
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
            for source in processed_data[date].keys():
                if source not in source_counts.keys():
                    source_counts[source] = []
                source_counts[source].append(processed_data[date][source]['total count'][target])

        source_count_list.append(source_counts)

    output = [targets, date_list, source_count_list]
    outfile.write(str(output))
    outfile.close()
    return output


def graphing(input_list: list, target_words_filename: str, sourcename: str):
    '''
        (list, str, str) -> png image files

        DESCRIPTION HERE

        Input: list from dataframe_setup in format [[targets], [dates], [{sources:[counts]}]]
        Output: None and graphs in .png format
    '''

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
            rows[i].append(input_list[2][i][sourcename][j])
    
    for i in range(len(rows)):
        rows[i].insert(0, input_list[1][i])

    with open(outfile, "w", newline='') as outfile:
        wr = csv.writer(outfile)
        wr.writerow(row_1)
        for line in rows:
            wr.writerow(line)

    return None


def main(input_data_filename: str, target_words_filename: str, sourcename: str):
    print("\nTOTAL TRACE START: {}".format(tt.ctime()))    
    main_start = tt.time()

    print("\n\t{:25} {}".format('TIME POPULATION START:', tt.ctime()))
    time_pop_start = tt.time()
    processed_data = populate_data(input_data_filename, target_words_filename, sourcename)
    print("\t{:25} {:.2f}s".format('TIME POPULATION RUNTIME:', runtime(time_pop_start)))

    print("\n\t{:25} {}".format('DATAFRAME SETUP START:', tt.ctime()))
    dataframe_start = tt.time()
    dataframe = dataframe_setup(processed_data, sourcename)
    print("\t{:25} {:.2f}μs".format('DATAFRAME SETUP RUNTIME:', 1000000*runtime(dataframe_start)))
    
    print("\n\t{:25} {}".format('CSV OUTPUT START:', tt.ctime()))
    csv_start = tt.time()
    csv_output(dataframe, sourcename)
    print("\t{:25} {:.2f}μs".format('CSV OUTPUT RUNTIME:', 1000000*runtime(csv_start)))

    # print("\n\tGRAPHING START: ", tt.ctime())
    # graph_start = tt.time()
    # graphing(dataframe, target_words_filename, sourcename)
    # print("\tGRAPHING RUNTIME: {:.2f}".format(runtime(graph_start)))

    print("\nTOTAL TRACE RUNTIME: {:.2f}".format(runtime(main_start)))

    return None

main('../aylien_data.jsonl', '../cluster1_coronavirus.txt', 'canada.ca')
