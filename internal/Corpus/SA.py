from transformers import pipeline
import tensorflow as tf

class SA:
  pipeline = None
  
  def __init__(self, model: str):
    self.pipeline = pipeline("text-classification", model=model)
    
  def getPrediction(self, text: str) -> int:
    """
    Returns the prediction for a document
    """
    
    result = self.pipeline(text)
        
    return result[0]["label"]