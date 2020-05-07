import json


count = 0
limit = 1
lines = []

with open('aylien_data.jsonl') as datastream:
    for i in range(30):
        lines.append(datastream.readline())

print("\n\n",json.loads(lines[0]))
print("\n",json.loads(lines[0]).keys())
# print("\n", (json.loads(lines[8])['published_at']))
print('\n\nDONE')