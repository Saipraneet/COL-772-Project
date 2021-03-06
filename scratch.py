from sense import SenseEmbedding;
from wordvec import WordEmbedding;
from l2c import L2CEmbedding;
from l2c_fast import L2CFastEmbedding
from keras.models import Sequential;
from keras.utils.np_utils import to_categorical
import cPickle
from keras.utils.generic_utils import Progbar
from keras.optimizers import Adagrad
import theano
from scipy import spatial, stats
from keras.preprocessing import text, sequence
import logging
from evaluate import Evaluate

import os, re, json
import random
from keras.utils import np_utils, generic_utils
import numpy as np

from sample_skipgrams import skipgrams_wordvec, skipgrams_sense, skipgrams_l2c, skipgrams_l2c_fast



################################################
# PARSING THE INPUT DATASET                    #
################################################


data_path = "wikipedia-dump/text8"
html_tags = re.compile(r'<.*?>')
to_replace = [('&#x27;', "'")]
hex_tags = re.compile(r'&.*?;')

def clean_comment(comment):
    c = str(comment.encode("utf-8"))
    c = html_tags.sub(' ', c)
    for tag, char in to_replace:
        c = c.replace(tag, char)
        c = hex_tags.sub(' ', c)
    return c


def text_generator(path=data_path):
    f = open(path)
    for i, l in enumerate(f):
        # comment_data = json.loads(l)
        # comment_text = comment_data["comment_text"]
        comment_text = clean_comment(l)
        if (i % 50000) == 100:
            # break
            print i
        yield comment_text
    f.close()


################################################
# EVALUATION OF THE LEARNT EMBEDDINGS          #
################################################
test_data_path = "test-data/ratings.txt"
f = open(test_data_path, 'r')
words = []
context = []
average_scores = []
for line in f:
    ele = line.split("\t")
    words.append((ele[1].lower(), ele[3].lower()))
    context.append((ele[5], ele[6]))
    average_scores.append(float(ele[7]))

print(len(words))
f.close()

average_scores = np.array(average_scores)


if __name__ == "__main__":
    model = Sequential()
    vocab_size = 50000
    dim = 300
    context_size = 4
    num_senses = 3
    nb_epoch = 2 

    model.add(L2CFastEmbedding(input_dim = 2*context_size + 2, vocab_dim = vocab_size+1, vector_dim = dim, num_senses = 3))
    optimizerObj = Adagrad(lr = 0.025)
    model.compile(loss="binary_crossentropy", optimizer= optimizerObj)
    fit = 1
    tokenizer_fname = "wikipedia_tokenizer_sense.pkl"
    if fit:
        print("Fit tokenizer...")
        tokenizer = text.Tokenizer(nb_words=vocab_size)
        tokenizer.fit_on_texts(text_generator())
        print("Save tokenizer...")
        f = open(tokenizer_fname, "wb")
        cPickle.dump(tokenizer, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    else:
        print('Load tokenizer...')
        f = open(tokenizer_fname, "rb")
        tokenizer = cPickle.load(f)
        f.close()

    evaluator = Evaluate(tokenizer, words, context, average_scores) 
    sampling_table = sequence.make_sampling_table(vocab_size)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)

        progbar = Progbar(tokenizer.document_count)
        samples_seen = 0
        losses = []
        batch_loss = []
        for i, seq in enumerate(tokenizer.texts_to_sequences_generator(text_generator())):
            # get skipgram couples for one text in the dataset
            couples, labels = skipgrams_l2c_fast(seq, vocab_size, num_senses =num_senses, window_size=4, negative_samples=1., sampling_table=sampling_table)
            if couples:
                # one gradient update per sentence (one sentence = a few 1000s of word couples)
                # print couples
                X = np.array(couples, dtype="int32")
                labels= np.array(labels, dtype="int32")
                loss = model.train_on_batch(X, labels)
                losses.append(loss)
                batch_loss.append(loss)
                if len(losses) % 10 == 0:
                    print ('\nBatch Loss: '+str(np.mean(batch_loss)))
                    progbar.update(i, values=[("loss", np.mean(losses))])
                    batch_loss = []
                samples_seen += len(labels)

                if (i and i % 10000 == 0):
                    global_weights, sense_weights = model.layers[0].get_weights()[0]
                    avgSim, avgSimC = evaluator.get_scores(global_weights, sense_weights)
                    print("scores after %d epochs:")
                    print("\t avg-sim: %5.3f", avgSim)
                    print("\t global-sim: %5.3f", avgSimC)

        print('Samples seen:', samples_seen)




    # Save the model    

    print("Training completed!")
    json_string = model.to_json()
    open('sense_vectors_wiki_architecture_lr.json', 'w').write(json_string)
    model.save_weights('sense_vectors_wiki_weights_lr.h5')


