from flask import Flask, render_template, request
from clean_text_query import clean_text
from compute_tf_idf import compute_tf_idf
from collections import OrderedDict
from operator import itemgetter
import os

app = Flask(__name__)

@app.route('/TextSearch', methods=['GET', 'POST'])
def TextSearch():
    if request.method == "GET":
        return render_template('TextSearch.php')
    elif request.method == "POST":
        input_text = request.form['search']
        filtered_text = clean_text(input_text)
        # files = os.listdir('metadata')
        files = os.listdir('actual_metadata')
        tf_idf = compute_tf_idf(files, filtered_text)
        print(tf_idf)
        tf_idf = OrderedDict(sorted(tf_idf.items(), key=itemgetter(1)))
        print(tf_idf)
        return render_template('TextSearch.php', data=tf_idf)

@app.route('/')
def main():
	return render_template('index_app.php')

if __name__ == '__main__':
	app.run(debug = True)

# Ashwin. with! Punct 1234 & nuber000