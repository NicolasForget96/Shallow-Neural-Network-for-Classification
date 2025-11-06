import numpy as np

# #######################################################
#               Gradient Descent
# #######################################################

class GradientDescent:

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def update_weights(self, W, dJ_dW):
        return W - self.alpha * dJ_dW
    

# #######################################################
#               Adam
# #######################################################

