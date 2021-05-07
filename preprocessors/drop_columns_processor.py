class DropColumnsProcessor:

  def __init__(self, columns):
    self.columns = columns

  def process(self, data):
    return data.drop(
      self.columns,
      axis=1,
    )