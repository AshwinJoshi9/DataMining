from flask import Flask, render_template, request, redirect
from markupsafe import Markup
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from PIL import Image
import cv2
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import math
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text_string):
    text_string = text_string.lower()
    text_string = re.sub(r'[^\w\s]', '', text_string)   # Removes punctuations from test query.
    # text_string = re.sub(r'\b\d+\b', '', text_string)   # Removes numbers from test query. Works for: ashwin 124. Doesn't work for: Ashwin124.
    text_string = text_string.strip()   # Strip unwanted character
    stop_words = set(stopwords.words('english'))    # English stop words.
    tokens = word_tokenize(text_string)     # Tokenized input query.
    filtered_words = [word for word in tokens if word not in stop_words]    # Removing stop words from the input test query
    return filtered_words

def read_data(metadata_path, image_url_path, caption_data):
    image_url_df = pd.read_json(image_url_path)
    image_url_map = dict(zip(image_url_df['image_id'], image_url_df['url']))
    corpus = pd.read_csv(metadata_path)
    captions_df = pd.read_csv(caption_data)
    # print(captions_df.head())
    # print(captions_df.columns)
    return corpus, image_url_map, captions_df

app = Flask(__name__)
metadata = 'actual_metadata/my_metadata.csv'
image_url = 'actual_metadata/image_data.json'
caption_data = 'actual_metadata/images.csv'
corpus, image_url_map, captions_df = read_data(metadata_path=metadata, image_url_path=image_url, caption_data=caption_data)

def classifier_YOLO(img_path):
    # print('In YOLO : ', img_path)
    img = img_path
    config = "yolov3.cfg"
    wts = "yolov3.weights"
    cls = "yolov3.txt"

    if 'http' in img:
        response = requests.get(img)
        image = Image.open(BytesIO(response.content))
        image = np.asarray(image)
        # print(image.shape)
        Width, Height, Depth = image.shape
    else:
        image = cv2.imread(img)
        Width = image.shape[1]
        Height = image.shape[0]
    scale = 0.00392

    # read class names from text file
    classes = None
    with open(cls, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # generate different colors for different classes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # read pre-trained model and config file
    net = cv2.dnn.readNet(wts, config)
    # create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # function to get the output layer names
    # in the architecture
    def get_output_layers(net):
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return label

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    # fin_label=""
    predicted_labels = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # fin_label=fin_label+str(classes[class_ids[i]])
        predicted_label = draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        predicted_labels.append(predicted_label)
    # print('-' * 20, 'Predicted Label YOLO', '-' * 20)
    # print(predicted_labels)
    # display output image
    cv2.imwrite('static/images/Classifier_Output.jpg', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return 'static/images/Classifier_Output.jpg', image, predicted_labels

def read_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = np.asarray(image)
    image = cv2.resize(image, (32, 32))
    return image

def read_model():
    # Load trained CNN model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return loaded_model_json

def classifier_CNN_CIFAR(url):
    # url = r"cifar_horse.jpg"
    # url = r"cifar_ship.jpg"
    # url = "http://ashwinjoshi.uta.cloud/RecApp/cifar_horse.jpg"
    loaded_model_json = read_model()
    tf.keras.backend.clear_session()
    image = read_url(url)
    model = model_from_json(loaded_model_json)
    model.load_weights('model.h5')
    # print('Model loaded')
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    image = np.expand_dims(image, axis=0)
    # print('In CIFAR_model : ', image.shape)
    predictions = model.predict(image)
    indices = np.argmax(predictions, 1)
    # print(predictions)
    # print([labels[x] for x in indices])
    return ([labels[x] for x in indices])

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

@app.route('/Classifier', methods=['POST'])
def Classifier():
    if request.method == "POST":
        # Yolo logic here
        input_text = request.form['search']
        # print('input_text is: ', input_text)
        image_path, final_image, YOLO_label = classifier_YOLO(input_text)
        class_label = classifier_CNN_CIFAR(input_text)
        # cv2.imshow('Final_image', final_image)
    return render_template('ClassifierResult.php', imgUrl=image_path, YOLO_label=YOLO_label, CIFAR_url=input_text, CIFAR_label=class_label[0])

@app.route('/CaptionBase', methods=['POST'])
def CaptionBase():
    return render_template('CaptionBase.php')

@app.route('/ClassifierBase', methods=['POST'])
def ClassifierBase():
    return render_template('ClassifierBase.php')

@app.route('/TextSearchBlog', methods=['POST'])
def TextSearchBlog():
    return redirect('http://ashwinjoshi.uta.cloud/RecApp/TextSearchBlog.html')
    # return render_template('TextSearchBlog.php')

@app.route('/TextSearchBase', methods=['POST'])
def TextSearchBase():
    return render_template('TextSearchBase.php')

def highlightText(data, input_query):
    resultData = ""
    for word in data:
        if word in input_query:
            resultData = resultData + '<span style="background:white;">' + word + '</span> '
        else:
            resultData = resultData + word + ' '
    return resultData

@app.route('/CaptionSearchBlog', methods=['POST'])
def CaptionSearchBlog():
    return redirect('http://ashwinjoshi.uta.cloud/RecApp/CaptionSearchBlog.html')


@app.route('/CaptionSearch', methods=['POST'])
def CaptionSearch():
    if request.method == "POST":
        caption_text = request.form['search']
        # print('caption_text is: ', caption_text)
        filtered_caption_text = clean_text(caption_text)
        final_df, idf_calc_header = compute_tf_idf(captions_df, filtered_caption_text)
        final_df['url'] = captions_df['url']
        final_v1_df = pd.DataFrame(columns=final_df.columns)
        for col in final_df.columns:
            if col == 'description':
                description_data = final_df['description']
                tempList = []
                for text in description_data:
                    text = text.split(' ')
                    tempList.append(Markup(str(highlightText(text, filtered_caption_text))))
                final_v1_df[col] = tempList
            else:
                final_v1_df[col] = final_df[col]
        return render_template('CaptionSearchResult.php', data=final_v1_df, header=' '.join(filtered_caption_text),
                               idf_calc_header=idf_calc_header, corpus_length=str(len(captions_df)),
                               filtered_text=filtered_caption_text)

@app.route('/TextSearch', methods=['POST'])
def TextSearch():
    if request.method == "POST":
        input_text = request.form['search']
        filtered_text = clean_text(input_text)
        final_df, idf_calc_header = compute_tf_idf(corpus, filtered_text)
        final_url_list = []
        for img_id in final_df['image_id']:
            final_url_list.append(image_url_map[img_id])
        final_df['url'] = final_url_list
        final_v1_df = pd.DataFrame(columns=final_df.columns)
        for col in final_df.columns:
            if col == 'description':
                description_data = final_df['description']
                tempList = []
                for text in description_data:
                    text = text.split(' ')
                    tempList.append(Markup(str(highlightText(text, filtered_text))))
                final_v1_df[col] = tempList
            else:
                final_v1_df[col] = final_df[col]

        return render_template('TextSearchResult.php', data=final_v1_df, header=' '.join(filtered_text), idf_calc_header=idf_calc_header, corpus_length=str(len(corpus)), filtered_text=filtered_text)

@app.route('/')
def main():
    return render_template('index_app.php')

app.run(debug=True)