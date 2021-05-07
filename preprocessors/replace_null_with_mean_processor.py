import numpy as np

class ReplaceNullWithMeanProcessor:

  def __init__(self, feature):
    self.feature = feature

  def process(self, data):
    return np.where(
      data[self.feature].isnull(),
      data[self.feature].mean(),
      data[self.feature]
    )