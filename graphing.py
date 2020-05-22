import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import csv as csv
import time as tt
import dataset_processing as dp
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


def pull_multi_freq_data(target_word_file: str, input_data_file: str): ## IPR
    '''
        (str, str) -> [[str], [{str:[int]}]]
        Pulls frequency data for single word from processed dataset in two filenames returned from dataset_processing.py
        Returns in the format:
        [[date1, date2, date3, ... , daten], 
        [{source1: [word1day1_ct, word2day1_ct, ... , wordnday1_ct], source2: [word1day1_ct, word2day1_ct, ... , wordnday1_ct]},
        {source1: [word1day2_ct, word2day2_ct, ... , wordnday2_ct], source2: [word1day2_ct, word2day2_ct, ... , wordnday2_ct]}]]
    '''
    input_data = dp.main_process(input_data_file, target_word_file)
    date_list = []
    source_count_list = []
    output_file = open("output_data_stripped.txt", "w", encoding="utf-8")

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
    
    # for i in range(len(output[2])):
    #     print (output[2][i])
    #     print ('\n')

    output_file.write(str(output))
    output_file.close()
    # print('\n', output, '\n')
    return output


def graph_multi_term(target: str, input_list: list): ## IPR
    '''
        ([str], {str:[int]}]) -> graphs
    '''
    plt.close("all")
    # input_list[0].reverse()
    count_list = []
    outfile = open("counts_only.txt","w")

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
    data_in = pd.DataFrame(data=count_list, index=input_list[0], columns=input_list[1])
    ax = sb.lineplot(data=data_in, palette="tab10", linewidth=2.0)
    ax.set(xlabel='Date', ylabel='Occurrences')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

    outfile.write(str(input_list[0])+"\n")
    outfile.write(str(input_list[1])+"\n")
    outfile.write(str(count_list))
    outfile.close()
    counts_only = [input_list[0], input_list[1], count_list]

    return counts_only


def csv_output(output_filename: str, input_list: list):
    '''
        ([[dates],[target words],[[day1 counts],[day2 counts]]]) -> csv
    '''
    input_list[1].insert(0, 'Date')
    for i in range(len(input_list[0])):
            input_list[2][i].insert(0, input_list[0][i])

    with open(output_filename, "w", newline='') as outfile:
        wr = csv.writer(outfile)
        wr.writerow(input_list[1])
        
        for line in input_list[2]:
            wr.writerow(line)


def main_multi(target_word_file: str, input_data_file: str): ## IPR
    start_time = tt.time()
    print("\nSTART GRAPHING AT: {} \nRUNNING...".format(tt.ctime()))

    transfer = pull_multi_freq_data(target_word_file, input_data_file)
    counts = graph_multi_term('coronavirus', transfer)
    csv_output('csv_out.csv', counts)

    print("TOTAL", end=" ")
    dp.runtime(start_time)

####

# main_single('test_words.txt', 'aylien_data.jsonl')
main_multi('test_words.txt', 'aylien_data.jsonl')