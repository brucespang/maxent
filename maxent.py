import operator
import cmaxent
import gc

class MaxEnt:
    def __init__(self, classes, features=[]):
        self.classes = classes
        self.features = features
        self.weights = [[0]*len(features)]*len(classes)

    def train(self, data):
        self.weights = cmaxent.train(self, data)

    def predict(self, data):
        scores = cmaxent.likelihood(self, data)
        class_scores = zip(self.classes, scores)
        return max(class_scores, key=operator.itemgetter(1))[0]

def f(args):
    print "feature args:", args
    return 1

if __name__ == "__main__":
    import random

    features = []
    features += [lambda d: int(d == "s")]
    features += [lambda d: int(d == "h")]
    features += [lambda d: int(d == "n")]

    maxent = MaxEnt(classes=["s", "h", "n"], features=features)
    maxent.train([("s", "s")] + [("h", "h")] + [("n", "n")])
    print "weights: ", maxent.weights
    print "s", maxent.predict("s")
    print "h", maxent.predict("h")
    print "n", maxent.predict("n")
