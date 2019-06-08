import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
nltk.download()

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile  #deep learning package word similarity
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/Users/sunlingyu/Downloads/glove/glove.6B.50d.txt')
word2vec_glove_file = get_tmpfile('glove.6B.100d.word2vec.txt')
print(glove2word2vec(glove_file,word2vec_glove_file))

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
print(model.most_similar('obama'))
print(model.most_similar('banana'))  #最相似
print(model.most_similar(negative='banana'))

result = model.most_similar(positive=['woman','king'],negative=['man']) #positive是近义词中最相似的
print(result)
print('{}:{:.4f}'.format(*result[0])) # .4f是保留小数点后四位  (float)  result[0]是因为前面返回的是按probability的顺序排的


def analogy(x1, x2, y1):
      result = model.most_similar(positive=[y1, x2], negative=[x1])
      return result[0][0]
print(analogy('japan','japanese','china'))
print(analogy('australia', 'beer', 'uk'))

print(model.doesnt_match("breakfast cereal dinner lunch".split()))

def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]

    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)

display_pca_scatterplot(model,
                        ['coffee', 'tea', 'beer', 'wine', 'brandy', 'rum', 'champagne', 'water',
                         'spaghetti', 'borscht', 'hamburger', 'pizza', 'falafel', 'sushi', 'meatballs',
                         'dog', 'horse', 'cat', 'monkey', 'parrot', 'koala', 'lizard',
                         'frog', 'toad', 'monkey', 'ape', 'kangaroo', 'wombat', 'wolf',
                         'france', 'germany', 'hungary', 'luxembourg', 'australia', 'fiji', 'china',
                         'homework', 'assignment', 'problem', 'exam', 'test', 'class',
                         'school', 'college', 'university', 'institute'])
plt.show()

