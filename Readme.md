Development Phase 1:
  Text search to render relevant document ranking from the courpus.
  Algorithm used: Tf-idf statistical metrics as the similiarity metric to rank documents.
  
Purpose of this idea:
  1. Goal is to build an image caption model
  2. Having a large data size worth 7Gigs of metadata it is hard to explore all of it manually. Many of the jsons do not load in Notepad++
  3. I made use of this approach to try different test queries to explore all the documents in the corpus
  4. This gave me an idea of targeting appropriate documents for specific outputs from the classifier model to map Object -> Relationship -> Object mapping
  
How to use this:

Deployed stuff: ashwinapp.azurewebsites.net
  1. Go this url, and click on any of the three feature buttons:<br>
    - TextMining App<br>
    Enter any free form text and the app will render most relevant images for the user text along with algorithm calculation steps.<br>
    - Classifier App<br>
        Enter image url and the app will output the objects it detected from that image. (Trained for 10 classes)<br>
    - Image Caption App<br>
        WORKING ON THIS<br>
For development on top of this and ofcourse a local test run:
  1. Clone this directory to your local repo.
  2. Put your metadata in: actual_metadata/*<.json>
  3. Execute index.py which fires the flask api and renders an html page on the URL mentioned in your console
  4. Try the app locally on URL flashed on the console by the Flask server.
  
NOTE: Nothing fancy. This just leverages the power of document similarity to scope out or explore large metadata instead of exploring them manually cause if you try exploring manually you would end up with a system lag for loading such a large metadata.

Upcoming Development Phases:
  - Classifier Phase: 
        1. This would take images as input for the model
        2. Classify object based on 80 classes selected while training
        3. Leverage the power of Development Phase 1 to get relevant documents to look up for
  - Image Captioning:
        1. Try to gain information from the relevant documents to map Object -> Relationships -> Object and build a meaningful caption
