from maxent import MaxEnt
from nltk.corpus import names
import string
import random

class Gender:
    def letter_vowel(self, l):
        return l in ['a', 'e', 'i', 'o', 'u']

    def char_vowel(self, n):
        return lambda word: self.letter_vowel(word[n])

    def char_is(self, n, char):
        return lambda word: word[n] == char

    def length_above(self, n):
        return lambda word: len(word) > n

    def length(self, word):
        return len(word)

    def char_in(self, char):
        return lambda word: char in word

    def __init__(self):
        relations = [self.char_vowel(-1),
                     self.char_vowel(-2),
                     self.char_vowel(0),
                     self.char_vowel(1),
                     self.length,
                     self.length_above(4)]
        # for c in string.ascii_lowercase:
        #     relations.append(self.char_in(c))
        self.classifier = MaxEnt(classes=["male", "female"],
                                 relations=relations)
    def train(self, names):
        return self.classifier.train(names)

    def guess(self, word):
        return self.classifier.predict(word)

if __name__ == "__main__":
    gender = Gender()
    names = ([('male', name) for name in names.words('male.txt')] +
             [('female', name) for name in names.words('female.txt')])
    random.shuffle(names)
    train_set, test_set = names[500:1000], names[:500]
    gender.train(train_set)

    num_correct = 0
    for (g, name) in test_set:
        guess = gender.guess(name)
        print {"name": name, "guess": guess, "actual": g}
        if guess == g:
            num_correct += 1
    print "num correct: ", num_correct
    print "accuracy: ", float(num_correct)/float(len(test_set))
