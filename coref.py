import nltk
from nltk.corpus import wordnet
from maxent import MaxEnt
from gender import gender
import itertools
import operator

# from http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

def flatten(l):
    return [item for sublist in l for item in sublist]

class Mention:
    def __init__(self, tree):
        self.words = [l[0] for l in tree.leaves()]
        self.mention_type = tree.node

        genders = [(x, gender.guess(x)) for x in self.words]
        # strip out words unrelated to people's names
        self.genders = [g for g in genders if g[1] == 'male' or g[1] == 'female']
        self.majority_gender = most_common([g for (_,g) in self.genders])

        # Because why dependency parse when you can hack it?
        synset = wordnet.synsets(self.genders[0][0])
        self.senses = set(synset)
        self.hypernyms = set(flatten([x.hypernyms() for x in synset]))

class CorefFeatures:
    def mention_type(self, pair):
        pair[0].mention_type == pair[1].mention_type

    def num_matches(self, pair):
        count = 0
        for i in range(0, len(pair[0].words)):
            for j in range(0, len(pair[1].words)):
                count += int(pair[0].words[i] == pair[1].words[j])
        return count

    def num_substrings(self, pair):
        count = 0
        for i in range(0, len(pair[0].words)):
            for j in range(0, len(pair[1].words)):
                w1 = pair[0].words[i]
                w2 = pair[1].words[j]
                count += int(w1.find(w2) > 0 or w2.find(w1) > 0)
        return count

    def gender(self, pair):
        return pair[0].majority_gender == pair[1].majority_gender

    def synonyms(self, pair):
        return len(pair[0].senses.intersect(pair[1].senses)) > 0

    def hypernyms(self, pair):
        return len(pair[0].hypernyms.intersect(pair[1].hypernyms)) > 0

    def wordnet_similarity(self, pair):
        return synset(pair[0]).path_similarity(pair[1])

    def last_words(self, pair):
        return pair[0].words[-1] == pair[1].words[-1]

class Coreference:
    def __init__(self):
        feat = CorefFeatures()
        features = [feat.mention_type,
                    feat.num_matches,
                    feat.num_substrings,
                    feat.gender,
                    feat.synonyms,
                    feat.hypernyms,
                    feat.wordnet_similarity,
                    feat.last_words]

        self.classifier = MaxEnt(classes=["is-coref", "is-not-coref"],
                                 features=features)

    def extract_mentions(self, doc):
        sentences = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        poses = [nltk.pos_tag(ws) for ws in words]
        chunks = [nltk.ne_chunk(pos) for pos in poses]
        trees = []
        for c in chunks:
            for x in c:
                if isinstance(x, nltk.Tree):
                    trees.append(x)

        return [Mention(t) for t in trees]

    def train(self, doc):
        pass

if __name__ == "__main__":
    coref = Coreference()
    print coref.extract_mentions("San Francisco is foggy. Amherst is snowy. Bruce is happy.")
