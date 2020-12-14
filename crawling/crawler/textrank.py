from konlpy.tag import Komoran
from textrank import KeysentenceSummarizer

import csv
import re

def load_data(filename):
    f = open(filename, 'r')
    lines = csv.reader(f)
    sents = []
    for i, line in enumerate(lines):
        tmp = line[1].replace(". ", "\n")
        tmp = tmp.split("\n")
        tmp = [x.strip() for x in tmp if len(x) > 0]
        sents.append([line[0], tmp])
    return sents 

komoran = Komoran()
sents = load_data("./data/sample_output_2019_newline.csv")

def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words

summarizer = KeysentenceSummarizer(
    tokenize = komoran_tokenizer,
    min_sim = 0.3,
    verbose = False
)

outputf = open("data/preprocessed_output_2019_newline.csv", "w", encoding='utf-8')
wr = csv.writer(outputf)
wr.writerow(['input', 'target'])

bracket_pattern = r'\[.*\]'
paren_pattern = r'\(.*\)'
for i, sent in enumerate(sents):
    print ("{}/{}".format(i, len(sents)))
    try:
        preprocessed_output = [re.compile(bracket_pattern).sub("", x) for x in sent[1]]
        preprocessed_input = re.compile(bracket_pattern).sub("", sent[0])
        preprocessed_output = [re.compile(paren_pattern).sub("", x) for x in preprocessed_output]
        preprocessed_input = re.compile(paren_pattern).sub("", preprocessed_input)
        keysents = summarizer.summarize(preprocessed_output, topk=1)
        extractedSent = keysents[0][2]
        title = sent[0].replace('"', '')
        preprocessed_sent = extractedSent.replace('"', '')
        if (len(title) * len(preprocessed_sent) > 0):
            wr.writerow([title, preprocessed_sent])
    except:
        print ("error")
        

outputf.close()

    
    