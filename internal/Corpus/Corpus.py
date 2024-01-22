import datetime
import pandas as pd

# from transformers import TFBertForSequenceClassification, BertTokenizerFast

from .Document import Document
from .LDA import LDA
from .SA import SA

class Corpus:
  
  # List of documents with class Document
  documents: list[Document] = []
  _LDA: LDA = None
  _SA: SA = None
  
  def __init__(self, documents: list(tuple[str, str, datetime.datetime])):
    self.documents = [Document(raw=document[0], source=document[1], date=document[2]) for document in documents]
    self._SA = SA("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

  def initLDA(self, numTopics: int = 5, passes: int = 10, workers: int = 4):
    """
    Initialises the LDA model
    """
    
    self._LDA = LDA(documents=self.documents, numTopics=numTopics, passes=passes, workers=workers)
    self._LDA.setTopicForDocuments(self.documents)
    
  def trainLDA(self, passes: int, workers: int = 4):
    """
    LDAs the corpus
    """
    
    self._LDA.trainLDAMore()
    self._LDA.setTopicForDocuments(self.documents)
    
  def assessLDA(self) -> tuple[float, float]:
    """
    Returns the perplexity and coherence of the LDA model
    """
    
    return self._LDA.getPerplexity(), self._LDA.getCoherence(self.documents)
      
  def toDataframe (self) -> pd.DataFrame:
    """
    Returns a dataframe of documents
    """
    
    return pd.DataFrame([document.toDict() for document in self.documents])
  
  def getSentimentForText(self, text: str) -> int:
    """
    Returns the sentiment for a document
    """
    
    return self._SA.getPrediction(text)