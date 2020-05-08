import json

count = 0
limit = 1
lines = []

with open('aylien_data.jsonl') as datastream:
    for i in range(30):
        lines.append(datastream.readline())     # unwraps 30 lines of the JSONL datafile and appends to list as python dictionary object

print("\n\n",json.loads(lines[0]))      # prints first dictionary
print("\n",json.loads(lines[0]).keys())     # shows all keys in dictionary 
print('\n\nDONE')