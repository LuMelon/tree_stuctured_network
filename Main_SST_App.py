from utils.tree_loader import Tree
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

def load_tree_from_file(filename):
    def sentence2tree(sentence):
        tree = Tree()
        tree.Parse_Sentence2Tree(sentence)
        return tree
    fr = open(filename, encoding='utf-8')
    sentences = fr.readlines()
    return [sentence2tree(sentence) for sentence in sentences]

tree_train = load_tree_from_file('Res/SST/train.txt')
tree_test = load_tree_from_file('Res/SST/test.txt')
tree_dev = load_tree_from_file('Res/SST/dev.txt')

words = []
[[words.extend(tree.get_nodes_attr('data')) for tree in trees] for trees in [tree_train, tree_dev, tree_test] ]
corpus = list(set(words))
del words
words2id = {word:id for id, word in enumerate(corpus)}

gensim_model = KeyedVectors.load_word2vec_format('Res/GoogleNews-vectors-negative300.bin', binary=True)
word_matrx = torch.tensor([gensim_model[word].tolist() if gensim_model.__contains__(word) else np.random.rand(300).tolist() for word in corpus])


