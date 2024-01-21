from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
from transformers import pipeline
import tensorflow as tf

class BERT:
  model: TFBertForSequenceClassification = None
  tokeniser: BertTokenizer = None
  
  def __init__(self):
    self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    self.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    
  def getPrediction(self, text: str) -> int:
    """
    Returns the prediction for a document
    """
    
    tokens = self.tokeniser.encode(text, truncation=True, padding=True, return_tensors="tf")    
    prediction = self.model.predict(tokens)
    probabilities = tf.nn.softmax(prediction.logits, axis=-1)
    
    print(prediction)
    print(probabilities)
    
    return tf.argmax(probabilities, axis=-1).numpy()[0]