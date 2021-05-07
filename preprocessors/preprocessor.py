import code

class Preprocessor:

  def __init__(self, data, preprocessors):
    self.data = data.copy(deep=True)
    self.preprocessors = preprocessors
   


  def process(self):
    for preprocessor in self.preprocessors:
      self.data = preprocessor.process(self.data)
    return self.data