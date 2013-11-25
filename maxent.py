from scipy.optimize import minimize
from math import log,exp
import operator
import numpy

def memoize(f):
    cache = {}
    def func(*args):
        if args in cache:
            return cache[args]
        else:
            res = f(*args)
            cache[args] = res
            return res
    return func

def trace(x):
    print x
    return x

class MaxEnt:
    def create_feature(self, k, r):
        return lambda c,d: int(k == c and r(d))

    def __init__(self, classes, features=[], relations=[]):
        self.classes = classes
        for r in relations:
            for k in classes:
                features += [self.create_feature(k, r)]
        self.weights = [0]*len(features)
        self.features = [memoize(f) for f in features]

    def feature_sum(self, weights, c, d):
        return sum([w*f(c,d) for (f,w) in zip(self.features, weights)])

    def likelihood(self, weights, c, d):
        actuals = exp(self.feature_sum(weights, c, d))
        print "actuals:", actuals
        expecteds = sum([exp(self.feature_sum(weights, c=k, d=d)) for k in self.classes])
        print "expecteds:", expecteds
        l = float(actuals)/float(expecteds)
        print "likelihood:", l
        return l

    def log_likelihood(self, weights, data):
        ll = sum([log(self.likelihood(weights, c=c, d=d)) for (d,c) in data])
        print "ll:", ll
        return ll

    def empirical_count(self, f, data):
        emp = sum([f(c,d) for (d,c) in data])
        print "empirical:", emp
        return emp

    def predicted_count(self, f, weights, data):
        pred = sum([self.likelihood(weights, c, d)*f(c,d) for c in self.classes for (d,_) in data])
        print "predicted:", pred
        return pred

    def gradient(self, weights, data):
        grad = [self.empirical_count(f, data) - self.predicted_count(f, weights, data) for f in self.features]
        print grad
        return grad

    def train(self, data):
        func = lambda weights: -1*self.log_likelihood(weights, data)
        gradient = lambda weights: numpy.array([-1*x for x in self.gradient(weights, data)])

        result = minimize(fun=func, jac=gradient, x0=self.weights,
                          method="BFGS", options={"disp": True})
        print result.x
        self.weights = result.x
        print self.weights

    def predict(self, data):
        probs = {c:self.likelihood(self.weights, c, data) for c in self.classes}
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
