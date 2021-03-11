import torch, flair, numpy as np
from flair.models import SequenceTagger
flair.device = torch.device("cpu")


def dump_embedding(key, e):
    if hasattr(e, "embeddings") and e.embeddings == "glove":
        dump_glove(key+"_glove", e)
    else:
        dump_flair_embedding(key, e)

def dump_glove(key, e):
    d = e.precomputed_word_embeddings
    dump_dict(key+"_words", list(s.encode() for s in d.index2word))
    dump_array(key+"_vectors", d.vectors)

def dump_chars(key, chars):
    dump_array(key, np.array(list(b'\xff'.join(chars)),dtype="B"))

def dump_flair_embedding(key, e):
    key = key + ("_forward" if e.is_forward_lm else "_reverse")
    dump_chars(key+"_chars", e.lm.dictionary.idx2item)
    dump_tensor(key+"_encoder", e.lm.encoder.weight)
    dump_linear(key+"_decoder", e.lm.decoder)
    dump_lstm(key+"_rnn", e.lm.rnn)

def dump_lstm(key, l):
    dump_tensor(key+"_bias_hh", l.bias_hh_l0)
    dump_tensor(key+"_bias_ih", l.bias_ih_l0)
    dump_tensor(key+"_weight_hh", l.weight_hh_l0)
    dump_tensor(key+"_weight_ih", l.weight_ih_l0)

    if l.bidirectional:
        dump_tensor(key+"_bias_hh_reverse", l.bias_hh_l0_reverse)
        dump_tensor(key+"_bias_ih_reverse", l.bias_ih_l0_reverse)
        dump_tensor(key+"_weight_hh_reverse", l.weight_hh_l0_reverse)
        dump_tensor(key+"_weight_ih_reverse", l.weight_ih_l0_reverse)

def dump_linear(key, l):
    dump_tensor(key+"_bias", l.bias)
    dump_tensor(key+"_weight", l.weight)

def dump_tensor(key, t):
    dump_array(key, t.data.numpy())

def dump_array(key, a):
    out[key] = a

def dump_dict(key, d):
    dump_array(key, np.array(list(b"\xff".join(d)), dtype="B"))

tagger = SequenceTagger.load("/home/sebk/.flair/models/en-ner-fast-conll03-v0.4.pt")

out = {}
for e in tagger.embeddings.embeddings:
    dump_embedding("embeddings", e)

dump_linear("embedding2nn", tagger.embedding2nn)
dump_lstm("rnn", tagger.rnn)
dump_linear("linear", tagger.linear)
dump_dict("tag_dictionary", tagger.tag_dictionary.idx2item)
dump_tensor("transitions", tagger.transitions)

np.savez(open("model.npz", "wb"), **out)
