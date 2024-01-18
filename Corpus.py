import datetime

from Document import Document
import gensim
import pandas as pd

class Corpus:
  
  # List of documents with class Document
  documents: list[Document] = []
  ldaModel: gensim.models.LdaMulticore
  dictionary: gensim.corpora.Dictionary
  
  def __init__(self, documents: list(tuple[str, str, datetime.datetime])):
    self.documents = [Document(raw=document[0], source=document[1], date=document[2]) for document in documents]
    
  # Mutators
    
  def LDA (self, numTopics: int, passes: int, workers: int = 4):
    """
    LDAs the corpus
    """
    
    # Generate a dictionary
    self.dictionary = gensim.corpora.Dictionary([document.tokens for document in self.documents])
    
    # Convert to bag of words (BOW)
    bowCorpus = [self.dictionary.doc2bow(document.tokens) for document in self.documents]
    
    # Do the thing!
    ldaModel = gensim.models.LdaMulticore(bowCorpus, num_topics=numTopics, id2word=self.dictionary, passes=passes, workers=workers)
    
    self.ldaModel = ldaModel
    
    # Set the topic for each document
    for document in self.documents:
      document.topic = self.getTopicForDocument(document)
  
  def getTopics (self, numTopics: int) -> list:
    """
    Returns the top numTopics topics
    """
    
    return self.ldaModel.print_topics(num_topics=numTopics)
  
  def getTopicForDocument(self, document: Document) -> int:
    """
    Returns the most probable topic for a document.
    """
    # Doc tokens to BOW
    doc_bow = self.dictionary.doc2bow(document.tokens)

    topic_probabilities = self.ldaModel.get_document_topics(doc_bow)

    # Sort by prob
    if topic_probabilities:
      most_probable_topic = sorted(topic_probabilities, key=lambda x: x[1], reverse=True)[0][0]
      return most_probable_topic
    else:
      return None
  
  def toDataframe (self) -> pd.DataFrame:
    """
    Returns a dataframe of documents
    """
    
    return pd.DataFrame([document.toDict() for document in self.documents])
  
  # Accessors