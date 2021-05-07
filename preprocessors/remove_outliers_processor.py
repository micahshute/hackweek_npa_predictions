class RemoveOutliersProcessor:

  def __init__(self, feature, minval, maxval):
    self.feature = feature
    self.minval = minval
    self.maxval = maxval

  def process(self, data):
    data = data.where(data[self.feature] > self.minval)
    data = data.where(data[self.feature] < self.maxval)
    return data