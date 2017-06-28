import numpy as np

# syn1 = 2*np.random.random((4,1)) - 1
# syn0 = 2*np.random.random((3,4)) - 1
# for j in xrange(60000):
#     l1 = 1/(1+np.exp(-(np.dot(X, syn0))))
#     l2 = 1/(1+np.exp(-(np.dot(l1, syn1))))
#     l2_delta = (y-l2)*(l2*(1-l2))
#     l1_delta = l2_delta.dot(syn1.T)*(l1 * (1 - l1))
#     syn1 += l1.T.dot(l2_delta)
#     syn0 += X.T.dot(l1_delta)
def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_deriv(o):
    #o is the output of nodes
    return o*(1-o)

if __name__ == "__main__":
    np.random.seed(1)

    X = np.array([[0, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1]])
    y = np.array([[0, ],
                  [1, ],
                  [1, ],
                  [0, ]])

    w0 = 2*np.random.random((3, 1)) - 1

    for iter in xrange(10000):
        #forward propagation
        l0 = X
        l1 = sigmoid(np.dot(l0, w0))

        #loss
        l1_error = y - l1

        #multiply node-wise error by gradient
        l1_delta = l1_error * sigmoid_deriv(l1)

        #update weights
        w0 += np.dot(l0.T, l1_delta)
        # b0 += l1_delta

    print "Output after training:"
    print l1