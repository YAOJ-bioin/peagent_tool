"""Minimal scBasset model architecture needed for PEAgent prediction.

This file vendors the small model-construction subset of scBasset used to load
the existing PEAgent bottleneck-64 `.h5` weights. The implementation is adapted
from Calico's scBasset project:

    https://github.com/calico/scBasset

The upstream project is licensed under the BSD 3-Clause License. See
`THIRD_PARTY_NOTICES.md` in this repository for attribution.
"""

from __future__ import annotations

import numpy as np


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for PEAgent model prediction.") from exc
    return tf


def make_model(bottleneck_size, n_cells, seq_len=1344, show_summary=True):
    """Create the scBasset sequence-to-cell prediction model."""

    tf = _tf()

    sequence = tf.keras.Input(shape=(seq_len, 4), name="sequence")
    current = sequence
    current, reverse_bool = StochasticReverseComplement()(current)
    current = StochasticShift(3)(current)
    current = conv_block(current, filters=288, kernel_size=17, pool_size=3)
    current = conv_tower(
        current,
        filters_init=288,
        filters_mult=1.122,
        repeat=6,
        kernel_size=5,
        pool_size=2,
    )
    current = conv_block(current, filters=256, kernel_size=1)
    current = dense_block(current, flatten=True, units=bottleneck_size, dropout=0.2)
    current = GELU()(current)
    current = final(current, units=n_cells, activation="sigmoid")
    current = SwitchReverse()([current, reverse_bool])
    current = tf.keras.layers.Flatten()(current)
    model = tf.keras.Model(inputs=sequence, outputs=current)
    if show_summary:
        model.summary()
    return model


class GELU(_tf().keras.layers.Layer):
    def call(self, x):
        tf = _tf()
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x


class StochasticReverseComplement(_tf().keras.layers.Layer):
    """Stochastically reverse complement a one-hot encoded DNA sequence."""

    def call(self, seq_1hot, training=None):
        tf = _tf()
        if training:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            reverse_bool = tf.random.uniform(shape=[]) > 0.5
            src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
            return src_seq_1hot, reverse_bool
        return seq_1hot, tf.constant(False)


class SwitchReverse(_tf().keras.layers.Layer):
    """Reverse predictions if the inputs were reverse complemented."""

    def call(self, x_reverse):
        tf = _tf()
        x = x_reverse[0]
        reverse = x_reverse[1]

        xd = len(x.shape)
        if xd == 3:
            rev_axes = [1]
        elif xd == 4:
            rev_axes = [1, 2]
        else:
            raise ValueError("Cannot recognize SwitchReverse input dimensions %d." % xd)
        return tf.keras.backend.switch(reverse, tf.reverse(x, axis=rev_axes), x)


class StochasticShift(_tf().keras.layers.Layer):
    """Stochastically shift a one-hot encoded DNA sequence."""

    def __init__(self, shift_max=0, pad="uniform", **kwargs):
        super().__init__(**kwargs)
        tf = _tf()
        self.shift_max = shift_max
        self.augment_shifts = tf.range(-self.shift_max, self.shift_max + 1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        tf = _tf()
        if training:
            shift_i = tf.random.uniform(
                shape=[],
                minval=0,
                dtype=tf.int64,
                maxval=len(self.augment_shifts),
            )
            shift = tf.gather(self.augment_shifts, shift_i)
            return tf.cond(
                tf.not_equal(shift, 0),
                lambda: shift_sequence(seq_1hot, shift),
                lambda: seq_1hot,
            )
        return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({"shift_max": self.shift_max, "pad": self.pad})
        return config


def shift_sequence(seq, shift, pad_value=0.25):
    tf = _tf()
    if seq.shape.ndims != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.shape
    pad = pad_value * tf.ones_like(seq[:, 0 : tf.abs(shift), :])

    def _shift_right(_seq):
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(
        tf.greater(shift, 0),
        lambda: _shift_right(seq),
        lambda: _shift_left(seq),
    )
    sseq.set_shape(input_shape)
    return sseq


def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    activation="gelu",
    strides=1,
    dilation_rate=1,
    l2_scale=0,
    dropout=0,
    conv_type="standard",
    residual=False,
    pool_size=1,
    batch_norm=True,
    bn_momentum=0.90,
    bn_gamma=None,
    bn_type="standard",
    kernel_initializer="he_normal",
    padding="same",
):
    tf = _tf()
    current = inputs
    conv_layer = tf.keras.layers.SeparableConv1D if conv_type == "separable" else tf.keras.layers.Conv1D
    if filters is None:
        filters = inputs.shape[-1]

    current = GELU()(current) if activation == "gelu" else tf.keras.layers.ReLU()(current)
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        bn_layer = (
            tf.keras.layers.experimental.SyncBatchNormalization
            if bn_type == "sync"
            else tf.keras.layers.BatchNormalization
        )
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)
    if residual:
        current = tf.keras.layers.Add()([inputs, current])
    if pool_size > 1:
        current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(current)
    return current


def conv_tower(
    inputs,
    filters_init,
    filters_end=None,
    filters_mult=None,
    divisible_by=1,
    repeat=1,
    **kwargs,
):
    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    current = inputs
    rep_filters = filters_init
    if filters_mult is None:
        if filters_end is None:
            raise ValueError("filters_end is required when filters_mult is None")
        filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

    for _ in range(repeat):
        current = conv_block(current, filters=_round(rep_filters), **kwargs)
        rep_filters *= filters_mult
    return current


def dense_block(
    inputs,
    units=None,
    activation="gelu",
    flatten=False,
    dropout=0,
    l2_scale=0,
    l1_scale=0,
    residual=False,
    batch_norm=True,
    bn_momentum=0.90,
    bn_gamma=None,
    bn_type="standard",
    kernel_initializer="he_normal",
):
    tf = _tf()
    current = inputs
    if units is None:
        units = inputs.shape[-1]

    current = GELU()(current) if activation == "gelu" else tf.keras.layers.ReLU()(current)
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape((1, seq_len * seq_depth))(current)

    current = tf.keras.layers.Dense(
        units=units,
        use_bias=not batch_norm,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)

    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        bn_layer = (
            tf.keras.layers.experimental.SyncBatchNormalization
            if bn_type == "sync"
            else tf.keras.layers.BatchNormalization
        )
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)
    if residual:
        current = tf.keras.layers.Add()([inputs, current])
    return current


def final(
    inputs,
    units,
    activation="linear",
    flatten=False,
    kernel_initializer="he_normal",
    l2_scale=0,
    l1_scale=0,
):
    tf = _tf()
    current = inputs
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape((1, seq_len * seq_depth))(current)

    return tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)
