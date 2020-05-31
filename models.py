import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
from keras.models import Model, model_from_yaml
from keras.layers import *
from keras.activations import softmax

from data_utils import PAD_SENT_TOKEN
from data_utils import MAXLEN


class ElmoEmbeddingLayer(Layer):
    # ElmoEmbeddingLayer based on https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += tf.trainable_variables(
            scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def compute_elmo(self, x):
        emb = self.elmo(x,
                        as_dict=False,
                        signature='default')
        msk = K.not_equal(x, PAD_SENT_TOKEN)
        return tf.where(msk, emb, tf.zeros_like(emb))

    def call(self, inputs, mask=None):
        print(inputs.shape)
        sqz_inputs = tf.squeeze(K.cast(inputs, tf.string), axis=2)

        print(sqz_inputs.shape)
        embs = tf.map_fn(self.compute_elmo, sqz_inputs, dtype=tf.float32)

        print(embs.shape)
        return embs

    # def compute_mask(self, inputs, mask=None):
    #     return K.not_equal(inputs, PAD_SENT_TOKEN)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dimensions)


def softmax_wrapper(x):
    return softmax(x, axis=1)


def sum_attention(x):
    return K.sum(x, axis=1)


def permute(x):
    return tf.transpose(x, perm=[1, 0, 2])


def build_elmo_model_full(prompt, elmo_trainable=False, only_elmo=False, use_mask=True, lstm_units=100):
    maxlen = MAXLEN[prompt]
    elmo = ElmoEmbeddingLayer(trainable=elmo_trainable)

    input_text = Input(shape=(maxlen, 1), dtype=tf.string)
    #    batch_shape=(batch_size, maxlen, 1))
    # perm = Lambda(permute)(input_text)
    # perm = TimeDistributed(Masking(mask_value=''))(perm)
    # embedding = ElmoEmbeddingLayer(trainable=trainable)(input_text)
    # embedding = TimeDistributed(elmo)(perm)
    embedding = elmo(input_text)
    # embedding = Flatten()(embedding)
    # embedding = Lambda(permute)(embedding)
    if use_mask:
        embedding = Masking(mask_value=0.0)(embedding)
    if not only_elmo:
        H = LSTM(lstm_units, return_sequences=True, name='lstm')(embedding)
        A_hat = Dense(lstm_units, activation='tanh', name='Attention_mat')(H)
        a = Dense(1, use_bias=False, activation=softmax_wrapper,
                  name='Attention_vec')(A_hat)
        o = Dot(1, name='')([a, H])
        o = Flatten()(o)
        score = Dense(1, activation='sigmoid', name='sigmoid')(o)
        model = Model(inputs=input_text, outputs=score)
    else:
        model = Model(inputs=input_text, outputs=embedding)
    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()
    return model


def build_elmo_model(input_shape_tuple, dropout, lstm_units):

    inputs = Input(
        shape=(input_shape_tuple[0], input_shape_tuple[1]), name='inputs')
    input_dropout = Dropout(dropout, name='dropout')(inputs)
    H = LSTM(lstm_units, return_sequences=True, name='lstm')(input_dropout)
    A_hat = Dense(lstm_units, activation='tanh', name='Attention_mat')(H)
    a = Dense(1, use_bias=False, activation=softmax_wrapper,
              name='Attention_vec')(A_hat)
    o = Dot(1, name='')([a, H])
    o = Flatten()(o)
    score = Dense(1, activation='sigmoid', name='sigmoid')(o)
    model = Model(inputs=inputs, outputs=score)
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def build_elmo_model_old(input_shape_tuple, dropout, lstm_units):
    inputs = Input(
        shape=(input_shape_tuple[0], input_shape_tuple[1]), name='inputs')
    dropout = Dropout(dropout, name='dropout')(inputs)
    lstm = LSTM(lstm_units, return_sequences=True, name='lstm')(dropout)
    A = Dense(lstm_units, activation='tanh', name='Attention_mat')(lstm)
    alpha = Dense(1, use_bias=False, activation=None, name='Attention_vec')(A)
    alpha = Reshape((input_shape_tuple[0],))(alpha)
    alpha = Activation('softmax')(alpha)
    alpha_re = RepeatVector(lstm_units)(alpha)
    alpha_perm = Permute((2, 1))(alpha_re)
    attention_mul = Multiply()([lstm, alpha_perm])
    out = Lambda(sum_attention, output_shape=None)(attention_mul)
    out = Dense(1, activation='sigmoid', name='sigmoid')(out)
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def get_layer_out(model, layer_index, data):
    intermediate_model = Model(inputs=model.input,
                               outputs=model.get_layer(index=layer_index).output)
    layer_out = intermediate_model.predict(data)
    return layer_out


def get_intermediate_outputs(model, data, layer_indices, layer_names):
    outputs = dict()
    # layer_indices = [2,3,4,6,11]
    # layer_names = ['lstm','AttW','AttV','softmax','out']
    # layer_indices = [2, 6, 11]
    # layer_names = ['lstm', 'softmax', 'out']
    for i in range(len(layer_indices)):
        layer_out = get_layer_out(model, layer_indices[i], data)
        outputs[layer_names[i]] = layer_out
        del layer_out
    return outputs


def get_model(prompt, fold, show_summary=False):
    tf.clear_session()
    yaml_string = open(
        'architecture/elmo_lstm_fix_data_prompt_{}.yml'.format(prompt), 'r').read()
    model = model_from_yaml(yaml_string)
    model.load_weights(
        'weight/elmo_lstm_fix_data_prompt_{}_fold_{}.BEST.h5'.format(prompt, fold))
    if show_summary:
        model.summary()
    return model
