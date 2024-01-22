import gensim
from Document import Document

class LDA:
  _ldaModel: gensim.models.LdaMulticore
  _dictionary: gensim.corpora.Dictionary
  _bowCorpus = None
  _trainCount = 0
  
  def __init__(self, documents: list[Document], numTopics: int, passes: int, workers: int = 4):
    """
    LDAs the corpus
    """
    
    # Generate a dictionary
    self._dictionary = gensim.corpora.Dictionary([document.tokens for document in documents])
    
    # Convert to the BOW
    self._bowCorpus = [self._dictionary.doc2bow(document.tokens) for document in documents]
    
    # Do the thing!
    ldaModel = gensim.models.LdaMulticore(self._bowCorpus, num_topics=numTopics, id2word=self._dictionary, passes=passes, workers=workers)
    
    self._ldaModel = ldaModel
        
  # Mutators
    
  def resetLDA(self):
    """
    Resets the LDA model
    """
    self._ldaModel = None
    self._dictionary = None
    self._trainCount = 0
    
  def trainLDAMore(self):
    """
    Trains the LDA model more
    """    
    
    self._ldaModel.update(self._bowCorpus)
      
  def setTopicForDocuments(self, documents: list[Document]):
    """
    Sets the topic for each document
    """
    
    for document in documents:
      document.topic = self.getTopicForDocument(document)
      
  def getTopics(self, numTopics: int) -> list:
    """
    Returns the top numTopics topics
    """
    
    return self._ldaModel.print_topics(num_topics=numTopics)
  
  # Accessors
  
  def getPerplexity(self) -> float:
    """
    Returns the perplexity of the model
    """
    
    return self._ldaModel.log_perplexity(self._bowCorpus)
  
  def getCoherence(self, documents: list[Document]) -> float:
    """
    Returns the coherence of the model
    """
    
    coherenceModel = gensim.models.CoherenceModel(model=self._ldaModel, texts=[document.tokens for document in documents], dictionary=self._dictionary, coherence='c_v')
    return coherenceModel.get_coherence()
  
  def getTopicForDocument(self, document: Document) -> int:
    """
    Returns the most probable topic for a document.
    """
    # Doc tokens to BOW
    doc_bow = self._dictionary.doc2bow(document.tokens)

    topic_probabilities = self._ldaModel.get_document_topics(doc_bow)

    # Sort by prob
    if topic_probabilities:
      most_probable_topic = sorted(topic_probabilities, key=lambda x: x[1], reverse=True)[0][0]
      return most_probable_topic
    else:
      return None