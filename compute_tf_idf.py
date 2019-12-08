import pandas as pd
from collections import defaultdict
import json
from clean_text_query import clean_text
import numpy as np
import math

def compute_tf_idf(corpus, filtered_text):
    corpus_size = len(corpus)
    tf_idf_mat = np.zeros((len(corpus), len(filtered_text)))
    idf_mat = np.zeros((1, len(filtered_text)))
    tf_calc_super = []
    for idx_doc, doc in enumerate(corpus['my_data']):
        tf_calc = []
        len_doc_str = "Length of doc is: " + str(len(doc))
        tf_calc.append(len_doc_str)
        # tf_calc_str += ("-" * 15)
        for idx_term, term in enumerate(filtered_text):
            tf = doc.count(term) / len(doc)
            if tf > 0:
                idf_mat[0, idx_term] += 1
            tf_idf_mat[idx_doc, idx_term] = tf

            term_freq_str = "'" + term + "' frequency for doc is: " + str(doc.count(term))
            tf_score_term = "'" + term + "' tf-score for doc is: " + str(tf)
            tf_calc.append(term_freq_str)
            tf_calc.append(tf_score_term)
        tf_calc_super.append(tf_calc)

    idf_calc_header = []
    for i in range(len(filtered_text)):
        idf_score = math.log((corpus_size / idf_mat[0, i]), 10)
        tf_idf_score = tf_idf_mat[:, i] * idf_score
        tf_idf_mat[:, i] = tf_idf_score
        idf_term_str = "Term '" + filtered_text[i] + "' idf for corpus is: " + str(idf_mat[0, i])
        normalized_idf_str = "Term '" + filtered_text[i] + "' idf-score(corpus length / term idf) for corpus is: " + str(idf_score)
        idf_calc_header.append(idf_term_str)
        idf_calc_header.append(normalized_idf_str)

    final_df = pd.DataFrame(columns=['image_id', 'tf_idf_score', 'tf_calc', 'description'])
    final_df['image_id'] = corpus['image_id']
    final_df['description'] = corpus['my_data']
    tempList = []
    for i in range(len(tf_idf_mat)):
        tempList.append(sum(tf_idf_mat[i, :]))
    final_df['tf_idf_score'] = tempList
    final_df['tf_calc'] = tf_calc_super
    final_df.sort_values(by=['tf_idf_score'], inplace=True, ascending=False)
    SelectKTop = 10
    return final_df.head(SelectKTop), idf_calc_header