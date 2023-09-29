import sys

import numpy as np

sys.path.append("../")


def getRandomIndex(n, x):
    np.random.seed(0)
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index

def majority_voting(models, x):
    return models.predict(x)