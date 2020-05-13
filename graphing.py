import matplotlib.pyplot as plt
import numpy as np
import dataset_processing as dp
import time as tt
import workspace as ws

def pull_single_freq_data(target_word_file: str, input_data_file: str):
    '''
        (str, str) -> [[str], [{str:[int]}]]
        Pulls frequency data for single word from processed dataset in two filenames returned from dataset_processing.py
        Returns in the format:
        [[date1, date2, date3, ... , daten], {source1: [day1_ct, day2_ct, ... , dayn_ct], source2: [day1_ct, day2_ct, ... , dayn_ct]}]
    '''
    input_data = ws.main_test_process(input_data_file, target_word_file, 400000)
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


def pull_multi_freq_data(target_word_file: str, input_data_file: str): ## IPR
    '''
        (str, str) -> [[str], [{str:[int]}]]
        Pulls frequency data for single word from processed dataset in two filenames returned from dataset_processing.py
        Returns in the format:
        [[date1, date2, date3, ... , daten], 
        [{source1: [word1day1_ct, word2day1_ct, ... , wordnday1_ct], source2: [word1day1_ct, word2day1_ct, ... , wordnday1_ct]},
        {source1: [word1day2_ct, word2day2_ct, ... , wordnday2_ct], source2: [word1day2_ct, word2day2_ct, ... , wordnday2_ct]}]]
    '''
    input_data = ws.main_test_process(input_data_file, target_word_file, 80000)
    date_list = []
    source_count_list = []

    with open(target_word_file, "r") as target_file:
        target_list = (target_file.read().splitlines())
    
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
    
    for i in range(len(output[2])):
        print (output[2][i])
        print ('\n')

    # print('\n', output, '\n')
    return output


def graph_multi_term(target: str, input_list: list): ## IPR
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
    ws.runtime(start_time)


def main_multi(target_word_file: str, input_data_file: str): ## IPR
    start_time = tt.time()
    print("\nSTART GRAPHING AT: {} \nRUNNING...".format(tt.ctime()))

    transfer = pull_multi_freq_data(target_word_file, input_data_file)
    graph_multi_term('coronavirus', transfer)

    print('\n\nDONE')
    print("TOTAL", end=" ")
    ws.runtime(start_time)

####

main_single('test_words.txt', 'aylien_data.jsonl')
# main_multi('test_words.txt', 'aylien_data.jsonl')