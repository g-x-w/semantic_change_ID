import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import csv as csv
import time as tt
import string as stng
import dataset_processing as dp
import graphing as gr

with open('top_50.txt') as sources:
    source_list = sources.read().splitlines()
    for i in range(len(source_list)):
        source_list[i] = source_list[i].strip()

targets = ['cluster1_coronavirus.txt', 'cluster2_quarantine.txt', 'cluster3_fatality.txt']

print('IT BEGINS')
source_count = 0
for i in range(15):
    print("\nBeginning source {} of 15 at {}".format(source_count+1, tt.ctime()))
    cluster_counter = 0
    for j in range(3):
        print("\tComputing cluster {} of 3...".format(cluster_counter+1))
        gr.main_multi(targets[j],'aylien_data.jsonl', source_list[i])
        cluster_counter += 1
        print("\t\tCluster {} of 3 completed.".format(cluster_counter))
    source_count += 1
    print("Source {} of 15 completed at {}".format(source_count, tt.ctime()))