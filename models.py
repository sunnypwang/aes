import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
from keras.models import Model, model_from_yaml
from keras.layers import *
from keras.activations import softmax
from keras.initializers import Constant
from data_utils import PAD_SENT_TOKEN
from data_utils import MAXLEN


class ElmoEmbeddingLayer(Layer):
    # ElmoEmbeddingLayer based on https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
    def __init__(self, maxlen, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.maxlen = maxlen
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += tf.trainable_variables(
            scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def compute_elmo(self, x):

        msk = K.not_equal(x, PAD_SENT_TOKEN)    # (maxlen,)
        x = tf.boolean_mask(x, msk)             # (?, )
        emb = self.elmo(x,
                        as_dict=False,
                        signature='default')
        emb.set_shape((None, 1024))             # (?, 1024)
        s = tf.shape(emb)
        paddings = [[0, self.maxlen - s[0]], [0, 0]]
        # paddings = tf.Print(paddings, [paddings], '--- padding : ')
        pad = tf.pad(emb, paddings, 'CONSTANT', constant_values=0.)
        pad = tf.ensure_shape(pad, (self.maxlen, 1024))  # (maxlen, 1024)
        return pad

    def call(self, inputs, mask=None):
        print(inputs.shape)
        sqz_inputs = tf.squeeze(K.cast(inputs, tf.string), axis=2)
        embs = tf.map_fn(self.compute_elmo, sqz_inputs, dtype=tf.float32)
        return embs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dimensions)


def softmax_wrapper(x):
    return softmax(x, axis=1)


def sum_attention(x):
    return K.sum(x, axis=1)


def permute(x):
    return tf.transpose(x, perm=[1, 0, 2])


def build_elmo_model_full(prompt, elmo_trainable=False, only_elmo=False, use_mask=True, lstm_units=100, summary=True):
    maxlen = MAXLEN[prompt]
    elmo = ElmoEmbeddingLayer(maxlen, trainable=elmo_trainable)

    input_text = Input(shape=(maxlen, 1), dtype=tf.string)
    embedding = elmo(input_text)
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
    if summary:
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


def build_glove_model(prompt, vocab_size, emb_matrix, maxwords=50, emb_dim=50, drop_rate=0.2, glove_trainable=False, summary=True):
    maxlen = MAXLEN[prompt]

    input_word = Input(shape=(maxlen * maxwords,), dtype='int32')
    emb = Embedding(input_dim=vocab_size, output_dim=emb_dim,
                    trainable=glove_trainable, mask_zero=True, name='glove')(input_word)
    x = ZeroMaskedEntries()(emb)
    x = Dropout(drop_rate)(x)
    x = Reshape((maxlen, maxwords, emb_dim))(x)
    x = TimeDistributed(Convolution1D(
        filters=100, kernel_size=5, padding='valid'), name='zcnn')(x)
    x = TimeDistributed(FeiDongAttention(), name='avg_zcnn')(x)

    x = LSTM(units=100, return_sequences=True, name='hz_lstm')(x)
    x = FeiDongAttention(name='avg_hz_lstm')(x)
    score = Dense(1, activation='sigmoid',
                  name='output')(x)
    model = Model(input_word, score)
    model.compile(loss='mse', optimizer='rmsprop')
    if summary:
        model.summary()
    return model


class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


class FeiDongAttention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, return_attention=False, **kwargs):
        self.supports_masking = True
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        self.return_attention = return_attention
        super(FeiDongAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(
            input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(
            input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]

    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        weights = tf.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
        weights = K.softmax(weights)

        out = x * \
            K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        # print(out.shape)
        out_sum = K.sum(out, axis=1, keepdims=False)
        print(out_sum.shape)
        out_sum = K.cast(out_sum, K.floatx())
        if self.return_attention:
            return weights
        else:
            return out_sum

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return (input_shape[0], input_shape[1])
        else:
            return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    # def get_config(self):
    #     config = {'op': self.op, 'activation': self.activation,
    #               'init_stdev': self.init_stdev, 'return_attention': self.return_attention}
    #     base_config = super(FeiDongAttention, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))
