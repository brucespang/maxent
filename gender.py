from maxent import MaxEnt
from nltk.corpus import names
import random

class Gender:
    def letter_vowel(self, l):
        return l in ['a', 'e', 'i', 'o', 'u']

    def last_char_vowel(self, word):
        return self.letter_vowel(word[-1])

    def last_char_consonant(self, word):
        return not self.last_char_vowel(word)

    def first_char_vowel(self, word):
        return self.letter_vowel(word[0])

    def first_char_consonant(self, word):
        return not self.first_char_vowel(word)

    def length(self, word):
        return len(word)

    def __init__(self):
        relations = [self.last_char_vowel,
                     self.last_char_consonant,
                     self.first_char_consonant,
                     self.last_char_consonant,
                     self.length]
        self.classifier = MaxEnt(classes=["male", "female", "other"],
                                 relations=relations)
    def train(self, names):
        return self.classifier.train(names)

    def guess(self, word):
        return self.classifier.predict(word)

if __name__ == "__main__":
    gender = Gender()
    names = ([(name, 'male') for name in names.words('male.txt')] +
             [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(names)
    train_set, test_set = names[500:600], names[:500]
    gender.train(train_set)

    num_correct = 0
    for (name, g) in test_set:
        guess = gender.guess(name)
        print name, guess, g
        if guess == g:
            num_correct += 1
    print "num correct: ", num_correct
    print "accuracy: ", float(num_correct)/float(len(test_set))
