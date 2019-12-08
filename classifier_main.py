import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
# from matplotlib.pyplot import imshow
# import matplotlib.pyplot as plt
import cv2
import requests
from io import BytesIO

def classifier_YOLO(img_path):
    print('In YOLO : ', img_path)
    img = img_path
    config = "yolov3.cfg"
    wts = "yolov3.weights"
    cls = "yolov3.txt"

    if 'http' in img:
        response = requests.get(img)
        image = Image.open(BytesIO(response.content))
        image = np.asarray(image)
        print(image.shape)
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
    print('-' * 20, 'Predicted Label YOLO', '-' * 20)
    print(predicted_labels)
    # display output image
    cv2.imwrite('static/images/Classifier_Output.jpg', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return 'static/images/Classifier_Output.jpg', image, predicted_labels

def read_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    # image = Image.open(url)
    image = np.asarray(image)
    image = cv2.resize(image, (32, 32))
    # imshow(image)
    # plt.show()
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
    print('Model loaded')
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    image = np.expand_dims(image, axis=0)
    print('In CIFAR_model : ', image.shape)
    predictions = model.predict(image)
    indices = np.argmax(predictions, 1)
    print(predictions)
    print([labels[x] for x in indices])
    return ([labels[x] for x in indices])