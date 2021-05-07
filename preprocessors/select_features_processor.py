class SelectFeaturesProcessor:

  def __init__(self, features):
    self.features = features

  def process(self, data):
    return data[self.features]