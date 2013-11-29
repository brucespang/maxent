import operator
import cmaxent
import gc

class MaxEnt:
    def create_feature(self, k, r):
        return lambda c,d: int(k == c and r(d))

    def __init__(self, classes, features=[], relations=[]):
        self.classes = classes
        for r in relations:
            for k in classes:
                features += [self.create_feature(k, r)]
        self.weights = [0]*len(features)
        self.features = features

    def train(self, data):
        self.weights = cmaxent.train(self, data)
        print self.weights

    def predict(self, data):
        probs = {c:cmaxent.likelihood(self, (c, data)) for c in self.classes}
        # print probs
        return max(probs.items(), key=operator.itemgetter(1))[0]

if __name__ == "__main__":
    import random

    f1 = lambda c,d: int(c==d)
    f2 = lambda c,d: int(c!=d)
    maxent = MaxEnt(classes=["s", "h", "n"], features=[f1,f2])
    maxent.train([("s", "s")] + [("h", "h")])
    # maxent.train([(random.choice(["s", "h"]), random.choice(["s", "h"])) for _ in range(100)])
    print "s", maxent.predict("s")
    print "h", maxent.predict("h")
