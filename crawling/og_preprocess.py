import csv
import re

data_dir = 'merged.csv'

with open(data_dir, newline='') as read_file:
    reader = csv.DictReader(read_file)
    
    # This variable is a list of dictionary, which contains same value of reader.
    # However, the values are 'preprocessed' by the following processes. e.g. [{'title': ..., 'content': ...}]
    preprocessed_reader = []
    
    # (a) Preprocessing.
    for row in reader:
        # (a-1) Extracting key sentence from the content.
        content = row['inputs']
        title = row['targets']
        
        bracket_pattern = r'\s*\[.*\]\s*'
        paren_pattern = r'\s*\(.*\)\s*'

        left_bracket = r'\s*\['
        right_bracket = r'\]\s*'

        left_paren = r'\s*\('
        right_paren = r'\)\s*'
        
        preprocessed_content = re.compile(bracket_pattern).sub("", content)
        preprocessed_title = re.compile(bracket_pattern).sub("", title)
        
        preprocessed_content = re.compile(r"\*").sub("", preprocessed_content)
        preprocessed_title = re.compile(r"\*").sub("", preprocessed_title)
        
        preprocessed_title = re.compile(paren_pattern).sub("", preprocessed_title) 

        preprocessed_content = re.compile(left_bracket).sub(" ", preprocessed_content)
        preprocessed_title = re.compile(left_bracket).sub(" ", preprocessed_title)

        preprocessed_content = re.compile(right_bracket).sub(" ", preprocessed_content)
        preprocessed_title = re.compile(right_bracket).sub(" ", preprocessed_title)

        preprocessed_content = re.compile(left_paren).sub(" ", preprocessed_content)
        preprocessed_title = re.compile(left_paren).sub(" ", preprocessed_title)

        preprocessed_content = re.compile(right_paren).sub(" ", preprocessed_content)
        preprocessed_title = re.compile(right_paren).sub(" ", preprocessed_title)

        # Erase double spaces.
        preprocessed_content = re.compile(r"\s+").sub(" ", preprocessed_content)
        preprocessed_title = re.compile(r"\s+").sub(" ", preprocessed_title)

        # I think parenthesis includes meaningful infos.
        # preprocessed_content = re.compile(paren_pattern).sub("", preprocessed_content)
        
        preprocessed_reader.append({'': row[''], 'title': preprocessed_title,
                                    'content': preprocessed_content})
        
with open("modified_output_og.csv", 'w', newline='') as write_file:

    fieldnames = ['', 'inputs', 'targets']
    writer = csv.DictWriter(write_file, fieldnames=fieldnames)

    writer.writeheader()
    for i, row in enumerate(preprocessed_reader):
        if i%10000 == 0:
            print("10000 done.")
        writer.writerow({'': row[''], 'inputs': row['content'], 'targets': row['title']})