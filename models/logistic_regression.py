import numpy as np

class LogisticRegression:

  @classmethod
  def p(cls, x, beta):
    y_pred = np.matmul(x.T, beta)
    logistic_prob = 1 / (1 + np.exp(-y_pred))
    return logistic_prob

  @classmethod
  def logistic_iteration(cls, beta, lr, x_batch, y_batch):
    b = np.matrix(beta).T 
    xt = np.matrix(x_batch)
    y = np.matrix(y_batch)
    x = xt.T
    p = (1 / (1 + np.exp(-1 * b.T * xt))).T 
    bn = b - lr * (xt * (p-y))
    beta_next = bn.T.tolist()[0]
    cost = -(y * np.log(p) + (1-y) * np.log(1-p)).tolist()[0][0]
    return cost, beta_next

  def __init__(self, target, features):
    self.target = target
    self.features = features
    self.beta = np.random.randn(len(features)) 

  def train(self, train_dataset, targets, epochs=5000, learning_rate=0.00000000005):
    cost = 0
    for epoch in range(epochs):
      cost, beta_next = self.logistic_iteration(self.beta, learning_rate, train_dataset, targets)
      print('Epoch %3d, cost %.3f' % (epoch+1,cost))
      self.beta = beta_next
    return self.beta

  def train_until(self, train_dataset, targets, cost_value_goal, learning_rate=0.0000005, min_learning_rate=0.00000000000000000000000000000000000000000000005):
    epoch = 0
    cost_prev, beta_next = self.logistic_iteration(self.beta, learning_rate, train_dataset, targets)
    self.beta = beta_next
    cost = cost_prev
    while cost > cost_value_goal and learning_rate > min_learning_rate:
      cost_prev = cost
      cost, beta_next = self.logistic_iteration(self.beta, learning_rate, train_dataset, targets)
      print('Epoch %3d, cost %.3f' % (epoch + 1, cost))
      if cost < cost_prev:
        self.beta = beta_next
      else:
        learning_rate /= 10
      epoch += 1
    print(learning_rate)
    return self.beta

  def predict(self, prediction_data):
    return self.p(prediction_data, self.beta)
  
  def test(self, test_data):
    return self.p(prediciton_data, self.beta)
