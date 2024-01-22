import datetime
import pandas as pd
import pickle
import os

# from transformers import TFBertForSequenceClassification, BertTokenizerFast

from .Document import Document
from .LDA import LDA
from .SA import SA

class Corpus:
  
  # List of documents with class Document
  documents: list[Document] = []
  _LDA: LDA = None
  _SA: SA = None
  
  def __init__(self, documents: list(tuple[str, str, datetime.datetime]) = None, loadFromPath: str = None):
    
    # This is to stop the hugging face tokeniser from causing issues when copying the corpus
    # huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    # To disable this warning, you can either:
	  # - Avoid using `tokenizers` before the fork if possible
	  # - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Init SA
    self._SA = SA("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    
    # Init LDA
    if loadFromPath:
      self._load(loadFromPath)
      return
    
    self.documents = [Document(raw=document[0], source=document[1], date=document[2]) for document in documents]

  def initLDA(self, numTopics: int = 5, passes: int = 10, workers: int = 4):
    """
    Initialises the LDA model
    """
    
    self._LDA = LDA(documents=self.documents, numTopics=numTopics, passes=passes, workers=workers)
    # self._LDA.setTopicForDocuments(self.documents)
    
  def save(self, path: str):
    """
    Saves the corpus
    """
    
    self._LDA.saveModel(path)
    with open(path + ".documents", "wb") as file:
      pickle.dump(self.documents, file)
      
  def _load(self, path: str):
    """
    Loads the corpus
    """
    
    self._LDA = LDA(loadFromPath=path)
    with open(path + ".documents", "rb") as file:
      self.documents = pickle.load(file)
    
  def trainLDA(self):
    """
    LDAs the corpus
    """
    
    self._LDA.trainLDAMore()
    
  def setTopicForDocuments(self):
    """
    Sets the topic for each document
    """
    
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