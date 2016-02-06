import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  d_scores = np.zeros((num_train, num_classes)) 
  data_loss = 0.0

  s_exp = np.exp(X.dot(W))
  scores = s_exp / np.sum(s_exp,axis = 1).reshape(num_train, 1)

  for i in xrange(num_train):
    data_loss += -1.0 * np.log(scores[i][y[i]])
    for j in xrange(num_classes):
      if j == y[i]:
        d_scores[i][j] = scores[i][j] - 1.0
        continue
      else:
        d_scores[i][j] = scores[i][j]

  d_scores /= num_train
  reg_loss = 0.5 * reg * np.sum(W*W)
  data_loss /= num_train
  loss = data_loss + reg_loss

  dW = np.transpose(X).dot(d_scores) + reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  data_loss = 0.0

  s_exp = np.exp(X.dot(W))
  scores = s_exp / np.sum(s_exp,axis = 1).reshape(num_train, 1)

  data_loss = -1.0 * np.log(scores[np.arange(num_train), y]).sum() / num_train

  d_scores = scores
  d_scores[np.arange(num_train), y] -= 1.0
  d_scores /= num_train
  reg_loss = 0.5 * reg * np.sum(W*W)
  loss = data_loss + reg_loss

  dW = np.transpose(X).dot(d_scores) + reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

