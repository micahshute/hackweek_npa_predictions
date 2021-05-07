class AddNonlinearFeatureProcessor:

  def __init__(self, new_feature, nonlinear_lambda, feature):
    self.nonlinear_lambda = nonlinear_lambda
    self.feature = feature
    self.new_feature = new_feature

  def process(self, data):
    data[self.new_feature] = [self.nonlinear_lambda(val) for val in data[self.feature]]
    return data