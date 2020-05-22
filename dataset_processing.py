import json as js
import time as tt
import string as stng

'''
    Will output 1 python dictionary/JSON object object with format
        {
        date1: {
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
    in txt file and as JSON file. Sources are defined as single domain names (i.e., )
'''

def runtime(start):
    '''
        (float) -> float
        computes runtime
        start: float value of start time, computed in main() function
    '''
    end_time = tt.time()
    print('RUNTIME: ', end_time-start)


def time_data_populate(data_input_file: str):
    '''
        (str) -> {str:{str:{str:{str:int}}}} & txt file
        processes dataset and returns data object as triple-nested dictionary 
        as in script docstring
        only populates dates, sources
    '''
    time_pop_start = tt.time()
    output = open("target_data.txt","w", encoding="utf-8")
    output_dict = {}

    with open(data_input_file) as datastream:
        for line in datastream:
            js_obj = js.loads(line)
            date = (js_obj['published_at'].split())[0]
            time_published = (js_obj['published_at'].split())[1]
            domain = js_obj['source']['domain']
            article = js_obj['links']['permalink']

            if date not in output_dict.keys():
                output_dict[date] = {}

            if domain not in output_dict[date].keys():
                output_dict[date][domain] = {}

            if article not in output_dict[date][domain].keys():
                output_dict[date][domain][article] = {}

            if 'total count' not in output_dict[date][domain].keys():
                output_dict[date][domain]['total count'] = {}

            output_dict[date][domain][article]['Time Published'] = time_published

    output.write(str(output_dict))
    output.close()
    print("\nTIME POPULATION", end = " ")
    runtime(time_pop_start)
    return output_dict


def freq_data_populate(data_input_file: str, target_words: str, dict_input: dict):
    '''
        (str, str, dict) -> {str:{str:{str:{str:int}}}} & text file
        further processes output dictionary object from time_data_population
        populates sources with target terms and occurrence counts in source body text
    '''
    freq_pop_start = tt.time()
    output = open("output_data_full.txt","w", encoding="utf-8")
    targets = []

    with open(target_words) as datastream:
        for line in datastream.readlines():
            targets.append(line.strip())

    with open(data_input_file) as datastream:
        for line in datastream:
            js_obj = js.loads(line)
            date = (js_obj['published_at'].split())[0]
            domain = js_obj['source']['domain']
            article = js_obj['links']['permalink']

            body_list = (js_obj['body']).lower()
            body_list = body_list.translate(str.maketrans('','',stng.punctuation))
            body_list = body_list.split()

            for term in targets:
                if term not in dict_input[date][domain]['total count'].keys():
                    dict_input[date][domain]['total count'][term] = body_list.count(term)
                else:
                    dict_input[date][domain]['total count'][term] += body_list.count(term)
                dict_input[date][domain][article][term] = body_list.count(term)

    with open('target_data.json', "w") as json_out:
        js.dump(dict_input, json_out)

    output.write(str(dict_input))
    output.close()
    print("TARGET WORD POPULATION", end = " ")
    runtime(freq_pop_start)
    return dict_input


def main_process(dataset_filename: str, target_words_filename: str):
    '''
        (str, str) -> {str:{str:{str:{str:int}}}}
        Main processing function, takes string names of dataset and target word txt filenames
        Uses previous functions to populate output dictionary
    '''
    start_time = tt.time()
    print("\nSTART DATA PROCESSING AT: {} \nRUNNING...".format(tt.ctime()))

    test_dict = time_data_populate(dataset_filename)
    output_dict = freq_data_populate(dataset_filename, target_words_filename, test_dict)

    print('\n\nDONE')
    print("PROCESSING", end=" ")
    runtime(start_time)

    return output_dict

# main_process('aylien_data.jsonl', 'target_words.txt')