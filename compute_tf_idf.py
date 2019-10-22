import pandas as pd
from collections import defaultdict
import json
from clean_text_query import clean_text
import numpy as np

def read_file(path):
    print('path is: ', path)
    print('-' * 50)
    with open(path, 'r') as jsonObj:
        # file_json = json.load(jsonObj)
        file_json = jsonObj.read()
    jsonObj.close()
    # file_json = pd.read_json(path, lines=True)
    return file_json

def compute_tf_idf(files, filtered_text):
    tf_idf_dict = defaultdict(list)
    final_dict = {}
    corpus_size = len(files)
    for file_path in files:
        # path = 'metadata/' + file_path
        path = 'actual_metadata/' + file_path
        file = read_file(path)
        for term in filtered_text:
            tf_idf_dict[file_path].append(file.count(term))     # calculate tf per file  # {doc name: [tf1, tf2, tfn]}
        doc_count = 0
        for subList in tf_idf_dict.values():
            for elem in subList:
                if elem > 0:
                    doc_count += 1
            idf = doc_count / corpus_size
            temp = []
            for elem in subList:
                temp.append(elem * idf) # check here
            final_dict[file_path] = sum(temp)
    return final_dict