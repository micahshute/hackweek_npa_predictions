class DropNaProcessor:

  @classmethod
  def process(cls, data):
    data.dropna()
    return data