import numpy as np 

class TransformBooleanToNumericProcessor:

  def __init__(self, feature):
    self.feature = feature

  def process(self, data):
    data[self.feature] = np.where(data[self.feature] == "True", 1, 0)
    return data