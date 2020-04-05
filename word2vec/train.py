import itertools
import data
import model as w2vmodel
import torch
import csv
import numpy as np
import random

from nltk.metrics import spearman
from sklearn.metrics.pairwise import cosine_similarity as cosine

def batch(iterable, n):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=None)

def word_sim(model, word, comp):
    sim_nbor = 0
    with torch.no_grad():
        _, embed_word, embed_comp, _    = model.forward(word, comp, comp, only_embedding=False, context_is_word=True)
        _, embed_tag, embed_tag_comp, _ = model.forward(word, comp, comp, only_embedding=False, context_is_word=False)
        if len(embed_comp.shape) == 2:
            embed_word = embed_word.repeat(embed_comp.shape[0], 1)
            embed_tag  = embed_tag.repeat(embed_comp.shape[0], 1)
        
        sim_nbor = cosine(embed_word.squeeze().cpu().reshape(1, -1), embed_comp.squeeze().cpu().reshape(1, -1))
        sim_tag  = cosine(embed_tag.squeeze().cpu().reshape(1, -1), embed_tag_comp.squeeze().cpu().reshape(1, -1))

    return sim_nbor


def train(wikidata, model, device, batch_size, max_epochs, window_size=5, sample_limit=-1):
    #Q: silly. No shuffling is done.
    #Q: wikidataloader is a dataset. not loader. rename
    dataset = data.WikiDataLoader(
        wikidata, window_size, True, device, shuffle=True)
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=1)

    print("corpus size (w)", len(dataset.listcorpus))

    iter = 0
    display_freq = 100
    lr_start = 0.02 
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr_start)
    n_samples = 0
    val_every_x_batches = 30
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    for e in range(max_epochs):
        print("epoch", e)
        for word, nbor, negs, tag, t_nbor, t_negs in loader:

            optimizer.zero_grad()
            nbor = torch.stack(nbor, dim=1)
            negs = torch.stack(negs, dim=1)

            word = word.to(device)
            nbor = nbor.to(device)
            negs = negs.to(device)

            tag, t_nbor, t_negs = tag.to(device), t_nbor.to(device), t_negs.to(device)

            loss, _, _, _ = model.forward(word, nbor, negs)
            loss_tag, _, _, _ = model.forward(tag, t_nbor. t_negs)

            loss = loss/batch_size  #+ loss_tag / 2
            loss.backward()
            optimizer.step()

            if iter % display_freq == 0:
                print("sample batch loss", round(loss.item(), 5))
                print("progress:", round(100 * iter * batch_size / len(dataset.listcorpus), 4), "%")

            iter += 1
            if iter % val_every_x_batches == 0:
                test_wordsim(dataset, model, "wordsim353/combined.csv", device, silent=True)

            n_samples += batch_size
            if n_samples > sample_limit and sample_limit > 0:
                break

        if n_samples > sample_limit and sample_limit > 0:
            break

    torch.save(model.state_dict(), "latest.pth")
    return model


def test(wikidata, model, device, window_size, n_batches=30):
    dataset = data.WikiDataLoader(wikidata, window_size, False, device)
    #batch size 1 for helper calls
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=1)
    corr = 0
    ncorr = 0
    idx = 0
    val_every_x_batches = 30
    for word, nbor, negs, tag, t_nbor, t_negs in loader:
        nbor = torch.stack(nbor, dim=1).squeeze()
        negs = torch.stack(negs, dim=1).squeeze()

        word = word.to(device)
        nbor = nbor.to(device)
        negs = negs.to(device)

        sims_corr  = word_sim(model, word, nbor)
        sims_ncorr = word_sim(model, word, negs)

        corr_t, ncorr_t = sims_corr.sum().item(), sims_ncorr.sum().item()

        corr += corr_t
        ncorr += ncorr_t

    print("Correct score", corr, "Incorrect score", ncorr)


def test_wordsim(wikidata_loader, model, wordsim_csv, device, silent=False):
    # rows of Word 1, Word 2, Human (mean)
    # a,b,10.0
    data_raw = csv.DictReader(open(wordsim_csv))
    gt_pairs = {}
    pred_pairs = {}
    pairs_not_seen_n = 0
    for i, line in enumerate(data_raw):
        w1, w2, score = line['Word 1'], line['Word 2'], float(
            line['Human (mean)'])
        w1_id = wikidata_loader.wikidata.wikicorpus.dictionary.token2id.get(w1)
        w2_id = wikidata_loader.wikidata.wikicorpus.dictionary.token2id.get(w2)
        if w1_id == None or w2_id == None:
            pairs_not_seen_n += 1
            continue
        gt_pairs[(w1, w2)] = score

        w1_id = torch.LongTensor([w1_id]).to(device)
        w2_id = torch.LongTensor([w2_id]).to(device)
        pred = word_sim(model, w1_id, w2_id).sum()

        with torch.no_grad():
            _, embed_w1, _, _  = model.forward(w1_id, w1_id, w1_id, True)
            _, _, embed_w2, _  = model.forward(w2_id, w2_id, w2_id, True)

        pred_pairs[(w1, w2)] = pred.item()


    gt_ranks = {k: r for r, k in enumerate(
        sorted(gt_pairs,   key=gt_pairs.get,   reverse=True), 1)}
    pred_ranks = {k: r for r, k in enumerate(
        sorted(pred_pairs, key=pred_pairs.get, reverse=True), 1)}


    corr = spearman.spearman_correlation(gt_ranks, pred_ranks)

    from scipy.stats import spearmanr
    t = [(gt_pairs[k], pred_pairs[k]) for k in gt_pairs]
    corr2 = spearmanr([t[i][0] for i in range(len(t))], [t[i][1] for i in range(len(t))])
    print("#### CORR2", corr2)

    if not silent:
        print("GT PAIRS", gt_pairs, "\n")
        print("PRED PAIRS", pred_pairs, "\n")
        print("GT RANK", gt_ranks,"\n")
        print("PRED RANK", pred_ranks, "\n ### \n")
        print("correlation", corr, "\n ### \n")
        print("pairs removed", pairs_not_seen_n)
        print("pairs left", len(pred_pairs))


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

    #wikidata = data.Wikidata("simplewiki-latest-pages-articles.xml.bz2")
    wikidata = data.Wikidata("test-medium-wiki.xml.bz2")
    wikidata_loader = data.WikiDataLoader(wikidata, 8, False, device)
    #wikidata = data.Wikidata("test_small_wiki.xml.bz2") # "simplewiki-latest-pages-articles.xml.bz2")
    words_n = len(wikidata.wikicorpus.dictionary.token2id)
    model = w2vmodel.Word2VecBow(wikidata, words_n, device, 300, len(wikidata_loader.tag_to_idx))
    model = train(wikidata, model, device, 128, 1, 8) #5*10**6)
    #test(wikidata, model, device, 10)
    #test(wikidata, model, device, 8)
    test_wordsim(wikidata_loader, model, "wordsim353/combined.csv", device)
 