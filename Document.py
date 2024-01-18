import nltk
from nltk.corpus import stopwords

import string
import datetime

class Document:
  raw: str
  tokens: list[str]
  
  sentiment: int = None
  topic: int
  date: datetime.datetime
  source: str

  def __init__(self, raw: str, source: str, date: datetime.datetime):
    self.raw = raw
    self.source = source
    self.date = date
    
    self.preprocess(raw)
    
  def toDict(self) -> dict:
    """
    Returns a dictionary representation of the document
    """
    return {
      'raw': self.raw,
      'tokens': self.tokens,
      'topic': self.topic,
      'date': self.date,
      'source': self.source
    }

  def _clean(self, document: str) -> str:
    """
    Cleans a document by removing stopwords and punctuation
    """

    # Make lowercase
    document = document.lower()

    # Remove stopwords
    stopWords = set(stopwords.words('english'))
    document = ' '.join(
      [word for word in document.split() if word not in stopWords])
    # Remove punctuation
    document = ''.join(
      [char for char in document if char not in string.punctuation])

    return document

  def _tokenise(self, document: str) -> list:
    """
    Tokenises a document
    """
    return document.split()

  def _lemmatise(self, document: list) -> list:
    """
    Lemmatises a document
    """
    lemmatiser = nltk.WordNetLemmatizer()
    return [lemmatiser.lemmatize(word) for word in document]

  def _stem(self, document: list) -> list:
    """
    Stems a document
    """
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(word) for word in document]

  def preprocess(self, document: str):
    """
    Preprocesses a document
    """
    self.tokens = self._lemmatise(
      self._tokenise(
        self._clean(
          document
        )
      )
    )
    # self.stems = self._stem(document)