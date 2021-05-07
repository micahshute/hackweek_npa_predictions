import numpy as np

class TaxableAsScorpProcessor:

  @classmethod
  def process(cls, data):
    data["taxable_as_scorp"] = np.where(data["taxable_as_scorp"] == "True", 1, 0)
    return data