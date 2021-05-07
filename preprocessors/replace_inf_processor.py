import numpy as np

class ReplaceInfProcessor:

  @classmethod
  def process(cls, data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data