import torch     #nn #Torch text?
from torch.nn.functional import interpolate as ipolate
import numpy as np
import random
import nltk

import data

#x tensors, of size y. Vocabulary size -> embedding size

class Word2VecBow(torch.nn.Module):
    def __init__(self, wikidata, vocab_size, device, embed_size=100, tag_size=45):
        super(Word2VecBow, self).__init__()
        self.wikidata = wikidata
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed  = torch.nn.Embedding(vocab_size, embed_size, max_norm=1, norm_type=2, sparse=True).to(device)
        self.embed2 = torch.nn.Embedding(vocab_size, embed_size, max_norm=None, norm_type=2, sparse=True).to(device)
        self.tag_layer = torch.nn.Embedding(vocab_size, tag_size, max_norm=1, norm_type=2, sparse=True).to(device)
        self.tag_layer2 = torch.nn.Embedding(vocab_size, tag_size, max_norm=None, norm_type=2, sparse=True).to(device)

        torch.nn.init.uniform_(self.embed.weight, -self.embed_size/2 , self.embed_size/2)
        torch.nn.init.uniform_(self.embed2.weight, -0, 0)
        torch.nn.init.uniform_(self.tag_layer.weight, -tag_size/2 , tag_size/2)
        torch.nn.init.uniform_(self.tag_layer2.weight, -0, 0)


    def forward(self, word, nbor, negs, only_embedding=False, context_is_word=True):

        if context_is_word:
            layer_1 = self.embed
            layer_2 = self.embed2
        else:
            layer_1 = self.tag_layer
            layer_2 = self.tag_layer2

        embed_a = layer_1(word)

        #Q: special testing case. no window.
        #Layer 1 embedding and layer 2 embedding not necessarily same
        if only_embedding:
            embed_b = layer_1(nbor)
            embed_c = layer_1(negs)
            return 0, embed_a.squeeze(), embed_b.squeeze(), embed_c.squeeze()

        embed_b = layer_2(nbor)
        embed_c = layer_2(negs)

        if True:
            sim_1 = torch.bmm(embed_b, embed_a.unsqueeze(2)).squeeze(2)
            sim_2 = torch.bmm(embed_c, embed_a.unsqueeze(2)).squeeze()
        else: #leftover
            print(embed_a.shape)
            print(embed_b.shape)
            assert(False and "unreachable")
            #b, c: batch x window x feature
            #we calculate a nested batch dot product
            sim_1 = torch.bmm(embed_b.view(B, W, 1, F), embed_a.view(B, F, 1, 1)).reshape(B, F)
            sim_2 = torch.bmm(embed_c.view(B, W, 1, F), embed_a.view(B, F, 1, 1)).reshape(B, F)

        logsigs_b = torch.log(torch.sigmoid((sim_1) + 0.01))
        logsigs_c = torch.log(torch.sigmoid((-sim_2) + 0.01))
        
        #lazy debug
        print_r = random.randint(0, 700)
        if not print_r:
            print(sim_1.shape)
            print(sim_2.shape)
            print(logsigs_b[0])
            print(logsigs_c[0])
        
        loss = -logsigs_b.sum() - logsigs_c.sum()
        return loss, embed_a, embed_b, embed_c


    #def backward(self, loss):
    #    loss.backward


if __name__ == "__main__":
    wikidata = data.Wikidata("simplewiki-latest-pages-articles.xml.bz2")
    model = Word2VecBow(wikidata, wikidata.vocab_size, 500)