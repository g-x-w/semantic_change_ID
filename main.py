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

with open('tracing_sources.txt', "r") as sources:
    source_list = sources.read().splitlines()
    for i in range(len(source_list)):
        source_list[i] = source_list[i].strip()

targets_list = ['cluster1_coronavirus.txt', 'cluster2_quarantine_canada.txt', 'cluster3_fatality.txt']
targets = ['cluster2_quarantine_canada.txt']
num_targets = len(targets)

num_sources = len(source_list) # swap len(source_list) & number of sources as necessary

print("\n > > > TRACE BEGINS < < <")
print("TIME: {}".format(tt.ctime()))
trace_time = tt.time()
source_counter = 0

for i in range(num_sources):
    source_counter += 1
    print("\nBeginning source {} of {} at {}\tSource: {}".format(source_counter, num_sources, tt.ctime(), source_list[i]))
    cluster_counter = 0
    for j in range(num_targets):
        cluster_counter += 1
        print("\tComputing cluster {} of {}...".format(cluster_counter, num_targets))
        dp.main('aylien_data_july.jsonl', targets[j], source_list[i], 1)
        print("\tCluster {} of {} completed.\n".format(cluster_counter, num_targets))
    print("Source {} of {} completed at {}".format(source_counter, num_sources, tt.ctime()))
print("\n > > > TRACE CONCLUDED < < <")
print("TOTAL TIME: {} minutes".format((tt.time() - trace_time)/60))