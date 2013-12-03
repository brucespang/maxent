from maxent import MaxEnt
from nltk.corpus import names, gutenberg
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

    def is_vowel(self, n):
        return lambda word: n < len(word) and word[n] in self.vowels

    def contains(self, c):
        return lambda word: c in word.lower()

    def is_capital(self, n):
        return lambda word: n < len(word) and word[n].isupper()

    def __init__(self):
        features = [self.length,
                    self.num_vowels,
                    self.num_consonants,
                    self.vowel_ratio,
                    self.is_capital(0)]

        for c1 in string.ascii_lowercase:
            features.append(self.contains(c1))
            for c2 in string.ascii_lowercase:
                features.append(self.contains(c1 + c2))

        for i in range(0, 15):
            for c in string.ascii_lowercase:
                features.append(self.char_is(i, c))
            features.append(self.is_vowel(i))

        self.classifier = MaxEnt(classes=["male", "female", "other"],
                                 features=features)
    def train(self, names):
        return self.classifier.train(names)

    def guess(self, word):
        if word in ['she', 'her']:
            return 'female'
        elif word in ['he', 'him']:
            return 'male'
        else:
            return self.classifier.predict(word)

names = ([('male', name) for name in names.words('male.txt')] +
         [('female', name) for name in names.words('female.txt')])
names_set = set(names)
non_names = [('other', w) for w in gutenberg.words('austen-emma.txt') if w not in names_set and w not in ['she', 'her', 'he', 'him']]

words = names + non_names[:5000]

rand = random.Random(1)
# rand = random.Random()
rand.shuffle(words)
train_set, test_set = words[500:], words[:500]
gender = Gender()
try:
    weights = pickle.load(open("gender.pickle"))
    gender.classifier.weights = weights
except IOError:
    gender.train(train_set)
    pickle.dump(gender.classifier.weights, open("gender.pickle", 'w'))

if __name__ == "__main__":
    num_correct = 0
    for (g, w) in test_set:
        guess = gender.guess(w)
        if guess == g:
            num_correct += 1
        else:
            print {"word": w, "guess": guess, "actual": g}
    print "num correct: ", num_correct
    print "accuracy: ", float(num_correct)/float(len(test_set))
