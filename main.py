import json as js
import time as tt
import re as re
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import numpy as np
import seaborn as sb
import pandas as pd
import csv as csv
import dataset_processing_bysource as dp

with open('tracing_sources.txt') as sources:
    source_list = sources.read().splitlines()
    for i in range(len(source_list)):
        source_list[i] = source_list[i].strip()

targets = ['cluster1_coronavirus.txt', 'cluster2_quarantine.txt', 'cluster3_fatality.txt']

num_targets = len(targets)
num_sources = len(source_list)


print(" > > > TRACE BEGINS")
source_counter = 0
for i in range(len(source_list)): #len(source_list)
    print("\nBeginning source {} of {} at {}".format(source_counter+1, num_sources, tt.ctime()))
    cluster_counter = 0
    for j in range(3):
        print("\tComputing cluster {} of {}...".format(cluster_counter+1, num_targets))
        dp.main('aylien_data.jsonl', targets[j], source_list[i])
        cluster_counter += 1
        print("\tCluster {} of {} completed.".format(cluster_counter, num_targets))
    source_counter += 1
    print("Source {} of {} completed at {}".format(source_counter, num_sources, tt.ctime()))
print("\n > > > TRACE CONCLUDED")