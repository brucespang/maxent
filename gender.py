from maxent import MaxEnt
from nltk.corpus import names
import string
import pickle
import random

class Gender:
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']

    def char_is(self, n, char):
        return lambda word: n < len(word) and word[n] == char

    def length(self, word):
        return len(word)

    def char_in(self, char):
        return lambda word: char in word

    def letter_vowel(self, char):
        return char in self.vowels

    def num_vowels(self, word):
        return len([c for c in word if self.letter_vowel(c)])

    def num_consonants(self, word):
        return len(word) - self.num_vowels(word)

    def vowel_ratio(self, word):
        return self.num_vowels(word)/float(len(word))

    def __init__(self):
        features = [self.length,
                    self.num_vowels,
                    self.num_consonants,
                    self.vowel_ratio]
        # for i in range(0, 1):
        for c in self.vowels:
            features.append(self.char_is(-1, c))
        self.classifier = MaxEnt(classes=["male", "female"],
                                 features=features)
    def train(self, names):
        return self.classifier.train(names)

    def guess(self, word):
        return self.classifier.predict(word)

names = ([('male', name) for name in names.words('male.txt')] +
         [('female', name) for name in names.words('female.txt')])
rand = random.Random(1)
rand.shuffle(names)
train_set, test_set = names[100:], names[:10]
gender = Gender()
# try:
#     weights = pickle.load(open("gender.pickle"))
#     gender.classifier.weights = weights
# except IOError:
gender.train(train_set)
# pickle.dump(gender.classifier.weights, open("gender.pickle", 'w'))

if __name__ == "__main__":
    num_correct = 0
    for (g, name) in test_set:
        guess = gender.guess(name)
        print {"name": name, "guess": guess, "actual": g}
        if guess == g:
            num_correct += 1
    print "num correct: ", num_correct
    print "accuracy: ", float(num_correct)/float(len(test_set))
