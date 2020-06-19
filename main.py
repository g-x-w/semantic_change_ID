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

print(" > > > TRACE BEGINS")
source_count = 20
for i in range(10):
    print("\nBeginning source {} of 6 at {}".format(source_count+1, tt.ctime()))
    cluster_counter = 0
    for j in range(3):
        print("\tComputing cluster {} of 3...".format(cluster_counter+1))
        dp.main('aylien_data.jsonl', targets[j], source_list[i])
        cluster_counter += 1
        print("\tCluster {} of 3 completed.".format(cluster_counter))
    source_count += 1
    print("Source {} of 6 completed at {}".format(source_count, tt.ctime()))
print("\n > > > TRACE CONCLUDED")