'''
import os, glob
import pandas as pd

l = ['train', 'eval', 'dev', 'test']
for i, val in enumerate(l):
    merge = glob.glob(os.path.join(".", val + "*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in merge)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged.to_csv("merged.csv")
'''

import csv

input_name = '\ufeffinputs'
dictionary = {'id':[], 'inputs':[], 'targets':[]}
with open('modified_output_og.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row)
        dictionary['id'].append(row[''])
        dictionary['inputs'].append(row['inputs'])
        dictionary['targets'].append(row['targets'])
        
with open('modified_output.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row)
        dictionary['id'].append(row[''])
        dictionary['inputs'].append(row['inputs'])
        dictionary['targets'].append(row['targets'])

with open('merged_cos.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['', 'inputs', 'targets'])
    
    writer.writeheader()
    length = len(dictionary['inputs'])
    for i in range(length):
        if i%10000 == 0:
            print("10000 data finished.")
        writer.writerow({'': dictionary['id'][i], 'inputs': dictionary['inputs'][i],\
            'targets': dictionary['targets'][i]})