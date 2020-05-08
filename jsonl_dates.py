import json as js
import time as tt

def runtime(start):
    end_time = tt.time()
    print('RUNTIME: ', end_time-start)

start_time = tt.time()
output = open("dates_list.txt","w")

with open('aylien_data.jsonl') as datastream:
    for line in datastream:
        output.write(js.loads(line)['published_at'])
        output.write('\n')
        # print(js.loads(line)['published_at'])

output.close()
print('\n\nDONE')
runtime(start_time)