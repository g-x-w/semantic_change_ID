import json as js
import time as tt

'''
    Used for pulling various values from the objects in the jsonl data file
'''

global key_type, output_name, test_range
key_type = 'body'
output_name = 'bodies.txt'
test_range = 1

def runtime(start):
    '''
        (float) -> float
        computes runtime
        start: float value of start time, computed in main() function
    '''
    end_time = tt.time()
    print('RUNTIME: ', end_time-start)

def build_test_list(filename: str):
    '''
        (str) -> list
        helper function to build test list for smaller samples
        filename: str of filename containing data
    '''
    lines = []
    with open(filename) as datastream:
        for i in range(test_range):
            lines.append(datastream.readline()) 
    return lines

def pull_all_values(filename: str):
    '''
        (str) -> txt file
        pulls all values for key from jsonl data
        filename: string name of filename containing data 
        key_type: global string containing target key
        output_name: global string containing output filename
    '''
    output = open(output_name,"w")
    with open(filename) as datastream:
        for line in datastream:
            js_obj = js.loads(line)
            try:
                output.write(js_obj[key_type])
                output.write('\n')
            except Exception:
                pass
    output.close()

def pull_n_values(filename: str):
    '''
        (str) -> txt file
        pulls all values for key from jsonl data
        filename: string name of filename containing data 
        key_type: global string containing target key
        output_name: global string containing output filename
    '''
    output = open(output_name,"w")
    lines = build_test_list(filename)

    for line in lines:
        js_obj = js.loads(line)
        try:
            output.write(js_obj[key_type])
            output.write('\n')
        except Exception:
            pass
    output.close()

def main():
    start_time = tt.time()
    print("RUNNING...")
    
    pull_n_values('aylien_data.jsonl')

    print('\n\nDONE')
    runtime(start_time)

main()