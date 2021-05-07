class AddColumnProcessor:

  def __init__(self, feature, feature_data):
    self.feature = feature
    self.feature_data = feature_data

  def process(self, data):
    data[self.feature] = self.feature_data
    return data