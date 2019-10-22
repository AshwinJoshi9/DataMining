Development Phase 1:
  Text search to render relevant document ranking from the courpus.
  Algorithm used: Tf-idf statistical metrics with cosine similarity as the similiarity metric to rank documents.
  
Purpose of this idea:
  1. Goal is to build an image caption model
  2. Having a large data size worth 7Gigs of metadata it is hard to explore all of it manually. Many of the jsons do not load in Notepad++
  3. I made use of this approach to try different test queries to explore all the documents in the corpus
  4. This gave me an idea of targeting appropriate documents for specific outputs from the classifier model to map Object -> Relationship -> Object mapping
  
How to use this:
  1. Clone this directory to your local repo.
  2. Put your metadata in: actual_metadata/*<.json>
  3. Execute index.py which fires the flask api and renders an html page on the URL mentioned in your console
  4. Enter a test text for which you want to scope out which documents are most relevant for that type of input
  5. Target those documents specifically for input text type you would come across later in your business logic
  
NOTE: Nothing fancy. This just leverages the power of document similarity to scope out or explore large metadata instead of exploring them manually cause if you try exploring manually you would end up with a system lag for loading such a large metadata.
