import datetime
import pandas as pd

from transformers import TFBertForSequenceClassification, BertTokenizerFast

from Document import Document
from LDA import LDA
from BERT import BERT

class Corpus:
  
  # List of documents with class Document
  documents: list[Document] = []
  LDA: LDA = None
  BERT: BERT = None
  
  
  def __init__(self, documents: list(tuple[str, str, datetime.datetime])):
    self.documents = [Document(raw=document[0], source=document[1], date=document[2]) for document in documents]
    self.LDA = LDA()
    self.BERT = BERT()
    
  def trainLDA(self, numTopics: int, passes: int, workers: int = 4):
    """
    LDAs the corpus
    """
    
    self.LDA.trainLDA(numTopics=numTopics, documents=self.documents, passes=passes, workers=workers)
      
  def toDataframe (self) -> pd.DataFrame:
    """
    Returns a dataframe of documents
    """
    
    return pd.DataFrame([document.toDict() for document in self.documents])
  
  def getSentimentForText(self, text: str) -> int:
    """
    Returns the sentiment for a document
    """
    
    return self.BERT.getPrediction(text)