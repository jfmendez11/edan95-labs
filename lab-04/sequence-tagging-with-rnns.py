from keras import models, layers
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from conll_dictorizer import CoNLLDictorizer
import matplotlib.pyplot as plt


def load(file):
    file = file
    embeddings = {}
    glove = open(file)
    for line in glove:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:])
        embeddings[word] = vector
    glove.close()
    embeddings_dict = embeddings
    embeded_words = sorted(list(embeddings_dict.keys()))
    return embeddings_dict


def cos_similarity(word, embeddings):
    result = ['']*5
    cos_sim = [-1]*5
    u = embeddings[word]
    for w in embeddings:
        v = embeddings[w]
        if word != w:
            cs = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0][0]
            if cs > min(cos_sim):
                i = cos_sim.index(min(cos_sim))
                result[i] = w
                cos_sim[i] = cs
    return result, cos_sim


BASE_DIR = '/Users/JuanFelipe/GitHub/edan95-labs/lab-04/'


def load_conll2003_en():
    train_file = BASE_DIR + 'NER-data/eng.train'
    dev_file = BASE_DIR + 'NER-data/eng.valid'
    test_file = BASE_DIR + 'NER-data/eng.test'
    col_names = ['form', 'ppos', 'pchunk', 'ner']
    tra_sentences = open(train_file, encoding='utf8').read().strip()
    d_sentences = open(dev_file, encoding='utf8').read().strip()
    t_sentences = open(test_file, encoding='utf8').read().strip()
    return tra_sentences, d_sentences, t_sentences, col_names


def words_and_ner_tags(sentence):
    x_words = []
    y_ner = []
    for w in sentence:
        x_words += [w['form'].lower()]
        y_ner += [w['ner']]
    return x_words, y_ner


def build_sequences(corpus_dict):
    X_mat, Y_mat = [], []
    for s in corpus_dict:
        x, y = words_and_ner_tags(s)
        X_mat += [x]
        Y_mat += [y]
    return X_mat, Y_mat

embedding_file = BASE_DIR + 'glove.6B.100d.txt'
embeddings_d = load(embedding_file)

print('Found', len(embeddings_d), 'word vectors.')

print('Calculating for table')
print(cos_similarity('table', embeddings_d))
print('Calculating for france')
print(cos_similarity('france', embeddings_d))
print('Calculating for sweden')
print(cos_similarity('sweden', embeddings_d))

print()

train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()
conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
train_dict = conll_dict.transform(train_sentences)
dev_dict = conll_dict.transform(test_sentences)
val_dict = conll_dict.transform(dev_sentences)
print(train_dict[0])
print(train_dict[1])

print()

print(words_and_ner_tags(train_dict[1]))

print()

X_t, Y_t = build_sequences(train_dict)
X_t_vector = []
Y_t_vector = []
for row in range(len(X_t)):
    for col in range(len(X_t[row])):
        X_t_vector.append(X_t[row][col])
        Y_t_vector.append(Y_t[row][col])


vocabulary = set(X_t_vector + list(embeddings_d.keys()))
print(len(vocabulary))

print()

idx_word = dict(enumerate(vocabulary, start=2))
word_idx = {v: k for k, v in idx_word.items()}
idx_ner = dict(enumerate(set(Y_t_vector), start=2))
idx_ner.update({0: 0, 1: '?'})
ner_idx = {v: k for k, v in idx_ner.items()}


print(idx_ner)
print(ner_idx)

embedding_matrix = np.random.random((len(vocabulary) + 2, 100))

for word in vocabulary:
    if word in embeddings_d:
        embedding_matrix[word_idx[word]] = embeddings_d[word]


X_t_idx, Y_t_idx = [[word_idx.get(X_t[row][col], 1) for col in range(len(X_t[row]))] for row in range(len(X_t))], [[ner_idx[Y_t[row][col]] for col in range(len(Y_t[row]))] for row in range(len(Y_t))]

X_idx_pad = pad_sequences(X_t_idx, maxlen=150)
Y_idx_pad = pad_sequences(Y_t_idx, maxlen=150)

print(X_idx_pad[1])
print(Y_idx_pad[1])

Y_pad_categorical = to_categorical(Y_idx_pad, num_classes=len(idx_ner))
# ----------------------------------------------------------------------------------------------------------------------
X_val, Y_val = build_sequences(val_dict)

X_val_idx, Y_val_idx = [[word_idx.get(X_val[row][col], 1) for col in range(len(X_val[row]))] for row in range(len(X_val))], [[ner_idx[Y_val[row][col]] for col in range(len(Y_val[row]))] for row in range(len(Y_val))]

X_idx_pad_val = pad_sequences(X_val_idx, maxlen=150)
Y_idx_pad_val = pad_sequences(Y_val_idx, maxlen=150)

Y_pad_categorical_val = to_categorical(Y_idx_pad_val, num_classes=len(idx_ner))
# ----------------------------------------------------------------------------------------------------------------------
print()

X_d, Y_d = build_sequences(dev_dict)

print('line 149:', len(X_d))

X_d_idx, Y_d_idx = [[word_idx.get(X_d[row][col], 1) for col in range(len(X_d[row]))] for row in range(len(X_d))], [[ner_idx[Y_d[row][col]] for col in range(len(Y_d[row]))] for row in range(len(Y_d))]

X_idx_pad_d = pad_sequences(X_d_idx, maxlen=150)
Y_idx_pad_d = pad_sequences(Y_d_idx, maxlen=150)

print(X_idx_pad_d[1])
print(Y_idx_pad_d[1])

Y_pad_categorical_d = to_categorical(Y_idx_pad_d, num_classes=len(idx_ner))

# ----------------------------------------------------------------------------------------------------------------------

model = models.Sequential()
model.add(layers.Embedding(len(vocabulary) + 2, 100, mask_zero=True, input_length=150))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = True
model.add(layers.Bidirectional(layers.SimpleRNN(100, return_sequences=True)))
model.add(layers.Dense(len(ner_idx), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.summary()

history_rnn = model.fit(X_idx_pad, Y_pad_categorical, batch_size=128, epochs=4, validation_data=(X_idx_pad_val, Y_pad_categorical_val))
'''
acc = history_rnn.history['acc']
val_acc = history_rnn.history['val_acc']
loss = history_rnn.history['loss']
val_loss = history_rnn.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
'''
test_loss_rnn, test_accuracy_rnn = model.evaluate(X_idx_pad_d, Y_pad_categorical_d)
print('Loss:', test_loss_rnn, 'Accuracy:', test_accuracy_rnn)

print()

prediction_prob_rnn = model.predict(X_idx_pad_d)
prediction_rnn = prediction_prob_rnn.argmax(axis=-1)

print(prediction_rnn[1])
print(Y_pad_categorical_d[1])

prediction_ner_rnn = [[idx_ner[idx] for idx in pred] for pred in prediction_rnn]

print('line 207:' ,prediction_ner_rnn[1])

prediction_unpadded_rnn = []
for i in range(len(prediction_ner_rnn)):
    row = []
    for j in range(len(prediction_ner_rnn[i])):
        if Y_pad_categorical_d[i][j][0] == 0:
            row.append(prediction_ner_rnn[i][j])
    prediction_unpadded_rnn.append(row)

output_rnn = open('output-rnn.txt','w+')

for i in range(len(X_d)):
    for j in range(len(X_d[i])):
        s = X_d[i][j] + ' ' + Y_d[i][j] + ' ' + prediction_unpadded_rnn[i][j] + '\n'
        output_rnn.write(s)
    output_rnn.write('\n')

output_rnn.close()

# ----------------------------------------------------------------------------------------------------------------------
model2 = models.Sequential()
model2.add(layers.Embedding(len(vocabulary) + 2, 100, mask_zero=True, input_length=150))
model2.layers[0].set_weights([embedding_matrix])
model2.layers[0].trainable = True
model2.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True, recurrent_dropout=0.25)))
model2.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True, recurrent_dropout=0.25)))
model2.add(layers.Dense(len(ner_idx), activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model2.summary()

history_lstm = model2.fit(X_idx_pad, Y_pad_categorical, batch_size=128, epochs=12, validation_data=(X_idx_pad_val, Y_pad_categorical_val))

acc = history_lstm.history['acc']
val_acc = history_lstm.history['val_acc']
loss = history_lstm.history['loss']
val_loss = history_lstm.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_loss_lstm, test_accuracy_lstm = model2.evaluate(X_idx_pad_d, Y_pad_categorical_d)
print('Loss:', test_loss_lstm, 'Accuracy:', test_accuracy_lstm)

print()

prediction_prob_lstm = model2.predict(X_idx_pad_d)
prediction_lstm = prediction_prob_lstm.argmax(axis=-1)

print(prediction_lstm[1])
print(Y_pad_categorical_d[1])

prediction_ner_lstm = [[idx_ner[idx] for idx in pred] for pred in prediction_lstm]

print('line 207:', prediction_ner_lstm[1])

prediction_unpadded_lstm = []
for i in range(len(prediction_ner_lstm)):
    row = []
    for j in range(len(prediction_ner_lstm[i])):
        if Y_pad_categorical_d[i][j][0] == 0:
            row.append(prediction_ner_lstm[i][j])
    prediction_unpadded_lstm.append(row)

output_rnn = open('output-lstm.txt','w+')

for i in range(len(X_d)):
    for j in range(len(X_d[i])):
        s = X_d[i][j] + ' ' + Y_d[i][j] + ' ' + prediction_unpadded_lstm[i][j] + '\n'
        output_rnn.write(s)
    output_rnn.write('\n')

output_rnn.close()
