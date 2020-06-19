import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import csv as csv
import time as tt
import string as stng
import dataset_processing_byline as dp
import workspace as ws

def pull_single_freq_data(target_word_file: str, input_data_file: str):
    '''
        (str, str) -> [[str], [{str:[int]}]]
        Pulls frequency data for single word from processed dataset in two filenames returned from dataset_processing.py
        Returns in the format:
        [[date1, date2, date3, ... , daten], {source1: [day1_ct, day2_ct, ... , dayn_ct], source2: [day1_ct, day2_ct, ... , dayn_ct]}]
    '''
    input_data = dp.main_process(input_data_file, target_word_file)
    date_list = []
    source_counts = {}

    with open(target_word_file, "r") as target_file:
        target = (target_file.read().splitlines())[0]
    
    for date in input_data.keys():
        date_list.append(date)
        
        for source in input_data[date].keys():
            if source not in source_counts.keys():
                source_counts[source] = []
                for i in range(len(date_list)-1):
                    source_counts[source].append(0)
            source_counts[source].append(input_data[date][source]['total count'][target])

        for recorded_source in source_counts.keys():
            if recorded_source not in input_data[date].keys():
                source_counts[recorded_source].append(0)
    
    output = [date_list, source_counts]
    # print (output)
    return output


def graph_single_term(target: str, input_list: list):
    '''
        ([str], {str:[int]}]) -> graphs
    '''
    input_list[0].reverse()
    count_list = []

    for j in range(len(input_list[0])):
        count_list.append(0)
        for source in input_list[1].keys():
            count_list[j] += input_list[1][source][j]

    count_list.reverse()
    plt.figure(1)
    plt.plot(input_list[0], count_list)
    plt.show()


def main_single(target_word_file: str, input_data_file: str):
    start_time = tt.time()
    print("\nSTART GRAPHING AT: {} \nRUNNING...".format(tt.ctime()))

    transfer = pull_single_freq_data(target_word_file, input_data_file)
    graph_single_term('coronavirus', transfer)

    print('\n\nDONE')
    print("TOTAL", end=" ")
    dp.runtime(start_time)


def pull_multi_freq_data(target_word_file: str, input_data_file: str, sourcename=False):
    '''
        (str, str) -> [[str], [{str:[int]}]]
        Pulls frequency data for target words from processed dataset in two filenames returned from dataset_processing.py
        Returns in the format:
        [[date1, date2, date3, ... , daten], [target1, target2, target3, ..., targetn],
        [{source1: [word1day1_ct, word2day1_ct, ... , wordnday1_ct], source2: [word1day1_ct, word2day1_ct, ... , wordnday1_ct]},
        {source1: [word1day2_ct, word2day2_ct, ... , wordnday2_ct], source2: [word1day2_ct, word2day2_ct, ... , wordnday2_ct]}]]
    '''
    if sourcename == False:
        file_out_name = 'all_source_tok_freq.txt'
    else:
        file_out_name = '{}_output_stripped.txt'.format(sourcename)

    input_data = ws.main_process(input_data_file, target_word_file, 528848, sourcename) ## changeline for testing
    date_list = []
    source_count_list = []
    output_file = open(file_out_name, "w", encoding="utf-8")

    with open(target_word_file, "r") as target_file:
        target_list = (target_file.read().splitlines())
        for i in range(len(target_list)):
            if target_list[-1] == '_':
                target_list[i] = target_list[i].strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation)) + '_'
            else:
                target_list[i] = target_list[i].strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation))
    
    for date in input_data.keys():
        date_list.append(date)
        source_counts = {}

        for target in target_list:
            for source in input_data[date].keys():
                if source not in source_counts.keys():
                    source_counts[source] = []
                source_counts[source].append(input_data[date][source]['total count'][target])

        source_count_list.append(source_counts)
    
    output = [date_list, target_list, source_count_list]
    output_file.write(str(output))
    output_file.close()
    return output


def graph_multi_term(input_list: list, target_filename: str, sourcename=False): ### CLEAN UP THIS FLAMING DUMPSTER
    '''
        ([str], {str:[int]}]) -> graphs
        Takes output from pull_multi_freq_data and uses it to make graph
    '''
    plt.close("all")
    count_list = []
    type_list = []
    title_tok = 'Token Frequency'
    title_type = 'Type Frequency'

    if sourcename == False:
        title = 'Type and Token Frequency from All Sources in Dataset'
        file_out_name = 'all_source_tok_freq.txt'
    else:
        title = 'Type and Token Frequency from {} for {}.png'.format(sourcename, target_filename[:-4])
        file_out_name = '{}_typeandtok_freq_{}'.format(sourcename, target_filename)

    outfile = open(file_out_name,"w")

    for j in range(len(input_list[0])):
        count_list.append([])

    for k in range(len(input_list[0])):
        for m in range(len(input_list[1])):
            count_list[k].append(0)

    for n in range(len(input_list[0])):
        for p in range(len(input_list[1])):
            for key_val in input_list[2][n].keys():
                count_list[n][p] += input_list[2][n][key_val][p]
    
    sb.set(style="whitegrid") 
    plt.subplot(122)
    data_in_tok = pd.DataFrame(data=count_list, index=input_list[0], columns=input_list[1])
    tok_freq = sb.lineplot(data=data_in_tok, dashes=False, palette="tab10", linewidth=2.0)
    tok_freq.set(xlabel='Date', ylabel='Occurrences')
    tok_freq.set_title(title_tok)
    plt.title(title_tok)
    plt.xticks(rotation=70)
    plt.tight_layout()

    for j in range(len(input_list[1])):
        type_list.append([])

    for k in range(len(input_list[1])):
        for m in range(len(input_list[0])):
            type_list[k].append(0)

    for n in range(len(input_list[0])):
        for key_val in input_list[2][n].keys():
            for p in range(len(input_list[1])):
                if input_list[2][n][key_val][p] > 0:
                    type_list[p][n] += 1

    # print(type_list)    ### ???? USE GROUPED BARCHARTS AND FIGURE IT OUT
    plt.subplot(121)
    data_in_type = pd.DataFrame(data=type_list, index=input_list[1], columns=input_list[0])
    type_freq = sb.barplot(data=data_in_type, ci=None)
    type_freq.set(xlabel='Date', ylabel='Number of Articles with Occurrence(s)')
    plt.title(title_type)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig(title, dpi=400)

    outfile.write(str(input_list[0])+"\n\n")
    outfile.write(str(input_list[1])+"\n\n")
    outfile.write(str(count_list)+'\n\n')
    outfile.write(str(type_list))
    outfile.close()
    counts_only = [input_list[0], input_list[1], count_list, type_list]

    return counts_only


def csv_output(input_list: list, sourcename=False):
    '''
        ([[dates],[target words],[[day1 counts],[day2 counts]]]) -> csv
    '''
    if sourcename == False:
        output_filename = "all_source_tok_freq.csv"
    else:
        output_filename = "{}_type_tok_freq.csv".format(sourcename)

    input_list[1].insert(0, 'Date')
    for i in range(len(input_list[0])):
            input_list[2][i].insert(0, input_list[0][i])

    with open(output_filename, "w", newline='') as outfile:
        wr = csv.writer(outfile)
        wr.writerow(input_list[1])
        
        for line in input_list[2]:
            wr.writerow(line)


def main_multi(target_word_file: str, input_data_file: str, sourcename=False):
    start_time = tt.time()
    print("\nSTART AT: {} \nRUNNING...".format(tt.ctime()))

    transfer = pull_multi_freq_data(target_word_file, input_data_file, sourcename)
    counts = graph_multi_term(transfer, target_word_file, sourcename)
    csv_output(counts, sourcename)

    print("\tTotal", end=" ")
    dp.runtime(start_time)

main_multi('cluster1_coronavirus.txt','aylien_data.jsonl', 'reuters.com')