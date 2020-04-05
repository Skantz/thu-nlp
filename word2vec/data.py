import gensim
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from random import randint
import random


#merge to next
class Wikidata:
    def __init__(self, wiki_xml_bz2):
        self.wikipath  = wiki_xml_bz2
        self.wikicorpus = gensim.corpora.WikiCorpus(wiki_xml_bz2)

#rename. not a loader
class WikiDataLoader(Dataset):
    """ Torch dataloader from gensim wikicorpus
        gensim is used for read, tokenization ... """
    def __init__(self, wikidata, window_size, train=True, device='cpu', shuffle=False, filter=True):
        self.phase_is_train = train
        self.wikidata = wikidata
        self.listcorpus = [] #stored as index, not word
        self.words_num = -1
        self.window_size = window_size
        self.negs_num  = self.window_size - 1
        self.device = device
        self.word_freq = {}
        self.tagcorpus = []  #stored as tag, not tag index
        self.tag_to_idx = {}

        for d in self.wikidata.wikicorpus.get_texts():
            doc_tags = []
            for w in d:
                try:
                    self.word_freq[wikidata.wikicorpus.dictionary.token2id.get(w)] += 1
                except KeyError:
                    self.word_freq[wikidata.wikicorpus.dictionary.token2id.get(w)] = 1
                #wasteful. avoids hardcode tags
                try:
                    doc_tags.append(self.tag_to_idx[w])
                except KeyError:
                    self.tag_to_idx[w] = len(self.tag_to_idx)
            
            self.listcorpus += d
            self.tagcorpus  += [nltk.pos_tag(w)[1] for w in d]

        self.words_num = len(self.wikidata.wikicorpus.dictionary)
        self.corpus_size = len(self.listcorpus)


    def idx_to_onehot(self, index):
        index = torch.FloatTensor(index)
        one_hot = torch.nn.functional.one_hot(index.to(torch.int64), self.words_num).to(self.device)
        return one_hot

    def __len__(self):
        return len(self.listcorpus)

    def corpus_idx_to_id(self, idx):
        word = self.listcorpus[idx]
        return self.wikidata.wikicorpus.dictionary.token2id.get(word)


    def get_triplet(self, sample_id, get_word=True):
        if get_word:
            lookup = self.corpus_idx_to_id
            corpus = self.listcorpus
        else:
            lookup = self.tag_to_idx
            corpus = self.tagcorpus

        idx_nbor = [(sample_id + offset) % len(corpus)
                    for offset in range(-self.window_size//2, self.window_size//2 + 1)
                    if offset != 0]
        idx_negs = [random.randint(0, len(corpus) - 1) for _ in range(self.window_size)]
        id_nbor  = [lookup(idx) for idx in idx_nbor]
        id_word  = [lookup(sample_id)]
        id_negs  = [lookup(idx) for idx in idx_negs]

        return id_word, id_nbor, id_negs

    def __getitem__(self, sample_id, filter_common=True):
        """ return positive examples from window and negative samples outside window """

        id_word, id_nbor, id_negs = self.get_triplet(sample_id, get_word=True)
        id_tag, id_tag_nbor, id_neg_tag = self.get_triplet(sample_id, get_words=False)
        #wrap
        return id_word, id_nbor, id_negs, id_tag, id_tag_nbor, id_neg_tag



if __name__ == "__main__":
    wikipath_bz2 = sys.argv[1]
    wikicorpus = WikiCorpus(wikipath_bz2) #Wikidata("simplewiki-latest-pages-articles.xml.bz2")
    wikicorpus.dictionary.save("wiki_dict.dict")
    MmCorpus.serialize("wiki_corpus.mm", wikicorpus)  #  File will be several GBs.
