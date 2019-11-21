from bpemb import BPEmb
from conllu import parse_incr
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
import pprint 
import time, sys, logging


print("TensorFlow Version", tf.__version__)
print('GPU Enabled:', tf.test.is_gpu_available())

params = {
  'train_path_ud': "pl_sz-ud-train.conllu",
  'test_path_ud': "pl_sz-ud-test.conllu",
  'train_path_ner': "nkjp-ner.conllu",
  'batch_size': 16,
  'num_samples': 4978,
  'rnn_units': 300,
  'dropout_rate': 0.2,
  'clip_norm': 5.0,
  'lr': 3e-4,
  'num_patience': 3,
  'pos_size': 100,
  'heads_size': 100,
  'token_len': 6,
  'sentence_len': 32,
  'embedding_size': 50
}

print('Loading BPEmb')
bpemb_pl = BPEmb(lang="pl", dim=params["embedding_size"])


def train_label_encoders_UD():
  le_upostag = preprocessing.LabelEncoder()
  le_deprel = preprocessing.LabelEncoder()

  # this method iterates CONLL files and outputs relevant fields
  # to generate Label Encoder IDs and compute various stats
  def data_generator_tokens(f_path, params, bpemb_pl):
    print('Reading', f_path)
    data_file = open(f_path, "r", encoding="utf-8")
    for tokenlist in parse_incr(data_file):
      orths = [t['form'] for t in tokenlist]
      lemmas = [t['lemma'] for t in tokenlist]
      upostags = [t['upostag'] for t in tokenlist]
      heads = [t['head'] for t in tokenlist]
      deprels = [t['deprel'] for t in tokenlist]

      assert len(orths) == len(lemmas) == len(upostags) == len(heads) == len(deprels)

      yield (orths, lemmas, (upostags, heads, deprels))

  toksizes = [];
  postags_all = [];
  deprels_all = [];
  sentence_lens = []
  nr = 0
  for conll_tuple in data_generator_tokens(params["train_path"], params, bpemb_pl):
    orths, lemmas, (upostags, heads, deprels) = conll_tuple[:]
    sentence_lens.append(len(orths))
    for orth in orths:
      nrtoks = len(bpemb_pl.encode_ids(orth))
      toksizes.append(nrtoks)
      if nr > 10: break
      nr += 1
    postags_all.extend(upostags)
    deprels_all.extend(deprels)

  le_upostag.fit(postags_all)
  le_deprel.fit(deprels_all)

  return le_upostag, le_deprel




# this method iterates CONLL files and outputs padded IDs
# it's intended as TensorFlow graph input
def data_generator_ids_UD(f_path, params, bpemb_pl, le_upostag, le_deprel):
  print('Reading', f_path)
  data_file = open(f_path, "r", encoding="utf-8")
  for tokenlist in parse_incr(data_file):

      orths = [ t['form'] for t in tokenlist ]
      lemmas = [ t['lemma'] for t in tokenlist ]
      upostags = [ t['upostag'] for t in tokenlist ]
      heads = [ t['head'] for t in tokenlist ]
      deprels = [ t['deprel'] for t in tokenlist ]

      assert len(orths) == len(lemmas) == len(upostags) == len(heads) == len(deprels)

      # padowanie tokenow do token_len:
      tokenids = [ bpemb_pl.encode_ids(orth) for orth in orths ]
      tokenids_padded = tf.keras.preprocessing.sequence.pad_sequences(
          tokenids, maxlen=params["token_len"], value=-1.0)
      """
      # uzupelnic tokenids_padded do sentence_len - jesli krotsze, wypelnij -1:
      while tokenids_padded.shape[0] < params["sentence_len"]:
        tokenids_padded = np.append(tokenids_padded, [[-1] * 6] , axis=0)
      # uzupelnic tokenids_padded do sentence_len - jesli dluzsze, przytnij:
      if tokenids_padded.shape[0] > params["sentence_len"]:
        tokenids_padded = tokenids_padded[:params["sentence_len"],]
      assert tokenids_padded.shape == (params["sentence_len"], params["token_len"])
      """
      upostags_ids = le_upostag.transform(upostags)
      #upostags_padded = tf.keras.preprocessing.sequence.pad_sequences(
      #    [upostags_ids], maxlen=params["sentence_len"], value=-1.0)

      heads = np.array([-1.0 if h is None else h for h in heads])
      #heads_padded =  tf.keras.preprocessing.sequence.pad_sequences(
      #    [heads], maxlen=params["sentence_len"], value=-1.0)

      deprels_ids = le_deprel.transform(deprels)
      #deprels_padded =  tf.keras.preprocessing.sequence.pad_sequences(
      #    [deprels_ids], maxlen=params["sentence_len"], value=-1.0)

      #print("tokenids_padded: "+str(tokenids_padded.shape)+" upostags_ids: "
      #    +str(upostags_ids.shape)+" heads: "+str(heads.shape)+" deprels_ids: "+str(deprels_ids.shape))
      assert tokenids_padded.shape[1]==params["token_len"]
      assert tokenids_padded.shape[0]==upostags_ids.shape[0]==heads.shape[0]==deprels_ids.shape[0]

      yield (tokenids_padded, (upostags_ids, heads, deprels_ids ))
      #yield (tokenids_padded, (upostags_padded, heads_padded, deprels_padded ))

def dataset_UD(is_training, params, bpemb_pl, le_upostag, le_deprel):
  _shapes = ([None, None], ([None, ], [None, ], [None, ]))
  _types = (tf.int32, (tf.int32, tf.int32, tf.int32))
  _pads = (-1, (-1, -1, -1))

  if is_training:
    ds = tf.data.Dataset.from_generator(
      lambda: data_generator_ids_UD(params['train_path_ud'], params, bpemb_pl, le_upostag, le_deprel),
      output_shapes=_shapes,
      output_types=_types, )
    ds = ds.shuffle(params['num_samples'])
    ds = ds.padded_batch(params['batch_size'], _shapes, _pads)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  else:
    ds = tf.data.Dataset.from_generator(
      lambda: data_generator_ids_UD(params['test_path_ud'], params, bpemb_pl, le_upostag, le_deprel),
      output_shapes=_shapes,
      output_types=_types, )
    ds = ds.padded_batch(params['batch_size'], _shapes, _pads)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  return ds

le_upostag, le_deprel = train_label_encoders_UD()
le_ner = train_label_encoders_NER()



