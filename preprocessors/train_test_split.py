import numpy as np

class TrainTestSplit:

  def __init__(self, train_fraction):
    self.train_fraction = train_fraction

  def process(self, data):
    msk = np.random.rand(len(data)) < self.train_fraction
    train_df = data[msk].copy(deep=True)
    test_df = data[~msk].copy(deep=True)
    return train_df, test_df
