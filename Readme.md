<h1>How to use this:</h1>
<h3>Overview</h3>
    - TextMining App<br>
    Enter any free form text and the app will render most relevant images for the user text along with algorithm calculation steps.<br>
    - Classifier App<br>
        Enter image url and the app will output the objects it detected from that image. (Trained for 10 classes)<br>
    - Image Caption App<br>
        Caption Search App where Tfidf is ussed to analyze the captions predicted<br>
Video Link: <a href="https://youtu.be/iq54Zlbikeg"> Click Here </a>

<h3>Local test and deploy</h3>
For development on top of this and ofcourse a local test run:
  1. Clone this directory to your local repo.
  2. Put your metadata in: actual_metadata/*<.json>
  3. Execute index.py which fires the flask api and renders an html page on the URL mentioned in your console
  4. Try the app locally on URL flashed on the console by the Flask server.
  5. Deploy
  
NOTE: Nothing fancy. This just leverages the power of document similarity to scope out or explore large metadata instead of exploring them manually cause if you try exploring manually you would end up with a system lag for loading such a large metadata.

<h1>Other Details:</h1>
<h3>Dataset</h3>
<a href=""https://visualgenome.org/>Click here to view the Visual Genome data</a>
<h3>References</h3>
1. <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf"> Tf-idf Wiki </a> <br>
2. <a href="http://flask.palletsprojects.com/en/1.1.x/"> Flask Docs </a> <br>
3. <a href="https://www.tensorflow.org/tutorials/images/cnn"> Tensorflow CNN </a><br>
4. <a href="https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb"> Image Captioning Model </a><br>
