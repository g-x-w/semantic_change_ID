import json as js
import time as tt
import string as stng

def runtime(start: float):
    '''
        (float) -> float
        computes runtime
        start: float value of start time, computed in main() function
    '''
    end_time = tt.time()
    print('RUNTIME: ', end_time-start)


def build_test_list(filename: str, test_range: int):
    '''
        (str) -> list
        helper function to build test list for smaller samples
        filename: str name of filename containing data
    '''
    lines = []
    with open(filename) as datastream:
        for i in range(test_range):
            lines.append(datastream.readline()) 
    return lines


def time_data_populate(filename: str, test_range: int, sourcename=False):
    time_pop_start = tt.time()
    output = open("workspace.txt","w", encoding="utf-8")    
    output_dict = {}
    lines = build_test_list(filename, test_range)

    for line in lines:
        js_obj = js.loads(line)
        date = (js_obj['published_at'].split())[0]
        time_published = (js_obj['published_at'].split())[1]
        domain = js_obj['source']['domain']
        article = js_obj['links']['permalink']

        if (sourcename != False) and (sourcename in domain):
            if date not in output_dict.keys():
                output_dict[date] = {}
            if domain not in output_dict[date].keys():
                output_dict[date][domain] = {}
            if article not in output_dict[date][domain].keys():
                output_dict[date][domain][article] = {}
            if 'total count' not in output_dict[date][domain].keys():
                output_dict[date][domain]['total count'] = {}
            output_dict[date][domain][article]['Time Published'] = time_published
        elif (sourcename == False):
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


def freq_data_populate(filename: str, target_words: str, dict_input: dict, test_range: int, sourcename=False):
    '''
    '''
    freq_pop_start = tt.time()
    output = open("workspace.txt","w", encoding="utf-8")    
    lines = build_test_list(filename, test_range)
    targets = []

    with open(target_words) as datastream:
        for line in datastream.readlines():
            if line.strip()[-1] == '_':
                if ' ' in line.strip():
                    target = line.strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation)).split()
                    target[-1] += '_'
                    targets.append(list(target))
                else:
                    target = line.strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation))
                    target += '_'
                    targets.append(target)
            else:
                if ' ' in line.strip():
                    targets.append(tuple(line.strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation)).split()))
                else:
                    targets.append(line.strip().lower().replace('-',' ').translate(str.maketrans('','', stng.punctuation)))

    for line in lines:
        js_obj = js.loads(line)
        date = (js_obj['published_at'].split())[0]
        domain = js_obj['source']['domain']
        article = js_obj['links']['permalink']
        
        if (sourcename != False) and (sourcename not in domain):
            pass

        else:
            body_list = (js_obj['body']).lower()
            body_list = body_list.replace('-', ' ')
            body_list = body_list.translate(str.maketrans('','',stng.punctuation))
            body_list = body_list.split()

            for term in targets:
                if type(term) == str: ## edit for fatality/ies
                    if term[-1] == '_':
                        if term not in dict_input[date][domain]['total count'].keys():
                            term = term.translate(str.maketrans('','',stng.punctuation))
                            count = 0
                            for item in body_list:
                                if term in item:
                                    count += 1
                            term = (term + '_')
                            dict_input[date][domain]['total count'][term] = count
                        else:
                            term = term.translate(str.maketrans('','',stng.punctuation))
                            count = 0
                            for item in body_list:
                                if term in item:
                                    count += 1
                            term = (term + '_')
                            dict_input[date][domain]['total count'][term] += count
                        dict_input[date][domain][article][term] = count
                    else:
                        if term not in dict_input[date][domain]['total count'].keys():
                            dict_input[date][domain]['total count'][term] = body_list.count(term)
                        else:
                            dict_input[date][domain]['total count'][term] += body_list.count(term)
                        dict_input[date][domain][article][term] = body_list.count(term)
                
                elif type(term) == list:
                    term[-1] = term[-1].translate(str.maketrans('','',stng.punctuation))
                    token_count = 0
                    for i in range(len(body_list)):
                        if body_list[i] == term[0]:
                            count = 0
                            try:
                                for j in range(len(term)-1):
                                    if body_list[i+j] == term[j]:
                                        count += 1
                                if count == len(term)-1 and term[-1] in body_list[i+len(term)-1]:
                                    token_count += 1
                            except IndexError:
                                pass
                    term[-1] += '_'
                    if ' '.join(term) not in dict_input[date][domain]['total count'].keys():
                        dict_input[date][domain]['total count'][' '.join(term)] = token_count
                    else:
                        dict_input[date][domain]['total count'][' '.join(term)] += token_count
                    dict_input[date][domain][article][' '.join(term)] = token_count
                
                elif type(term) == tuple:
                    token_count = 0
                    for i in range(len(body_list)):
                        if body_list[i] == term[0]:
                            count = 0
                            try:
                                for j in range(len(term)):
                                    if body_list[i+j] == term[j]:
                                        count += 1
                                if count == len(term):
                                    token_count += 1
                            except IndexError:
                                pass
                    if ' '.join(term) not in dict_input[date][domain]['total count'].keys():
                        dict_input[date][domain]['total count'][' '.join(term)] = token_count
                    else:
                        dict_input[date][domain]['total count'][' '.join(term)] += token_count
                    dict_input[date][domain][article][' '.join(term)] = token_count

    output.write(str(dict_input))
    output.close()
    print("TARGET WORD POPULATION", end = " ")
    runtime(freq_pop_start)
    return dict_input


def main_process(dataset_filename: str, target_words_filename: str, test_range: int, sourcename=False):
    '''
        (str, str) -> dict
        Main processing function, takes string names of dataset and target word txt filenames
        Uses previous functions to populate output dictionary
    '''
    start_time = tt.time()
    print("\nSTART PROCESSING TEST AT: {} \nRUNNING...".format(tt.ctime()))

    test_dict = time_data_populate(dataset_filename, test_range, sourcename)
    output_dict = freq_data_populate(dataset_filename, target_words_filename, test_dict, test_range, sourcename)

    print('\n\nDONE')
    print("PROCESSING", end=" ")
    runtime(start_time)

    return output_dict

# main_test_process('aylien_data.jsonl', 'target_words.txt', 10)