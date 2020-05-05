# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module contains a registry of dataset classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
from magenta.models.gansynth.lib import spectral_ops
from magenta.models.gansynth.lib import util
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import lookup as contrib_lookup

Counter = collections.Counter


class BaseDataset(object):
  """A base class for reading data from disk."""

  def __init__(self, config):
    self._train_data_path = util.expand_path(config['train_data_path'])

  def provide_one_hot_labels(self, batch_size):
    """Provides one-hot labels."""
    raise NotImplementedError

  def provide_dataset(self):
    """Provides audio dataset."""
    raise NotImplementedError

  def get_pitch_counts(self):
    """Returns a dictionary {pitch value (int): count (int)}."""
    raise NotImplementedError

  def get_pitches(self, num_samples):
    """Returns pitch_counter for num_samples for given dataset."""
    all_pitches = []
    pitch_counts = self.get_pitch_counts()
    for k, v in pitch_counts.items():
      all_pitches.extend([k]*v)
    sample_pitches = np.random.choice(all_pitches, num_samples)
    pitch_counter = Counter(sample_pitches)
    return pitch_counter


class NSynthTFRecordDataset(BaseDataset):
  """A dataset for reading NSynth from a TFRecord file."""

  def __init__(self, config):
      super(NSynthTFRecordDataset, self).__init__(config)
      self._train_meta_path = None
      if 'train_meta_path' in config and config['train_meta_path']:
          self._train_meta_path = util.expand_path(config['train_meta_path'])
      else:
          magic_meta_path = os.path.join(config['train_root_dir'], 'meta.json')
          if os.path.exists(magic_meta_path):
              self._train_meta_path = magic_meta_path
          
      self._instrument_sources = config['train_instrument_sources']
      self._min_pitch = config['train_min_pitch']
      self._max_pitch = config['train_max_pitch']
    
  def _get_dataset_from_path(self):
    dataset = tf.data.Dataset.list_files(self._train_data_path)
    dataset = dataset.apply(contrib_data.shuffle_and_repeat(buffer_size=1000))
    dataset = dataset.apply(
        contrib_data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=20, sloppy=True))
    return dataset

  def provide_one_hot_labels(self, batch_size):
    """Provides one hot labels."""
    pitch_counts = self.get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    counts = [pitch_counts[p] for p in pitches]
    indices = tf.reshape(
        tf.multinomial(tf.log([tf.to_float(counts)]), batch_size), [batch_size])
    pitch_one_hot_labels = tf.one_hot(indices, depth=len(pitches))

    instrument_family_count = 11 # TODO: Remove hard coding
    indices = tf.random.uniform([batch_size], minval=0, maxval=instrument_family_count, dtype=tf.int64)
    instrument_one_hot_labels = tf.one_hot(indices, depth=instrument_family_count)

    one_hot_labels = tf.concat([pitch_one_hot_labels, instrument_one_hot_labels], axis=1)
    return one_hot_labels, [pitch_one_hot_labels.shape[1].value, instrument_one_hot_labels.shape[1].value]

  def provide_dataset(self):
    """Provides dataset (audio, labels) of nsynth."""
    length = 64000
    channels = 1

    pitch_counts = self.get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    label_index_table = contrib_lookup.index_table_from_tensor(
        sorted(pitches), dtype=tf.int64)

    def _parse_nsynth(record):
      """Parsing function for NSynth dataset."""
      features = {
          'pitch': tf.FixedLenFeature([1], dtype=tf.int64),
          'audio': tf.FixedLenFeature([length], dtype=tf.float32),
          'qualities': tf.FixedLenFeature([10], dtype=tf.int64),
          'instrument_source': tf.FixedLenFeature([1], dtype=tf.int64),
          'instrument_family': tf.FixedLenFeature([1], dtype=tf.int64),
      }

      example = tf.parse_single_example(record, features)
      wave, label = example['audio'], example['pitch']
      wave = spectral_ops.crop_or_pad(wave[tf.newaxis, :, tf.newaxis],
                                      length,
                                      channels)[0]
      pitch_one_hot_label = tf.one_hot(
          label_index_table.lookup(label), depth=len(pitches))[0]
      #Todo remove hardcoded instrument family count
      instrument_one_hot_label = tf.one_hot(
          example['instrument_family'], depth=11)[0]
      one_hot_label = tf.concat([pitch_one_hot_label, instrument_one_hot_label], axis=0)
      return wave, one_hot_label, label, example['instrument_source']

    dataset = self._get_dataset_from_path()
    dataset = dataset.map(_parse_nsynth, num_parallel_calls=4)

    # Filter just specified instrument sources
    def _is_wanted_source(s):
      return tf.reduce_any(list(map(lambda q: tf.equal(s, q)[0], self._instrument_sources)))
    dataset = dataset.filter(lambda w, l, p, s: _is_wanted_source(s))
    # Filter just specified pitches
    dataset = dataset.filter(lambda w, l, p, s: tf.greater_equal(p, self._min_pitch)[0])
    dataset = dataset.filter(lambda w, l, p, s: tf.less_equal(p, self._max_pitch)[0])
    dataset = dataset.map(lambda w, l, p, s: (w, l))
    return dataset

  def get_pitch_counts(self):
    if self._train_meta_path:
      with open(self._train_meta_path, "r") as meta_fp:
        meta = json.load(meta_fp)
      pitch_counts = {}
      for name, note in meta.items():
        pitch = note["pitch"]
        if pitch in pitch_counts:
          pitch_counts[pitch] += 1
        else:
          pitch_counts[pitch] = 1
    else:
      pitch_counts = {
        24: 711,
        25: 720,
        26: 715,
        27: 725,
        28: 726,
        29: 723,
        30: 738,
        31: 829,
        32: 839,
        33: 840,
        34: 860,
        35: 870,
        36: 999,
        37: 1007,
        38: 1063,
        39: 1070,
        40: 1084,
        41: 1121,
        42: 1134,
        43: 1129,
        44: 1155,
        45: 1149,
        46: 1169,
        47: 1154,
        48: 1432,
        49: 1406,
        50: 1454,
        51: 1432,
        52: 1593,
        53: 1613,
        54: 1578,
        55: 1784,
        56: 1738,
        57: 1756,
        58: 1718,
        59: 1738,
        60: 1789,
        61: 1746,
        62: 1765,
        63: 1748,
        64: 1764,
        65: 1744,
        66: 1677,
        67: 1746,
        68: 1682,
        69: 1705,
        70: 1694,
        71: 1667,
        72: 1695,
        73: 1580,
        74: 1608,
        75: 1546,
        76: 1576,
        77: 1485,
        78: 1408,
        79: 1438,
        80: 1333,
        81: 1369,
        82: 1331,
        83: 1295,
        84: 1291
      }
    return pitch_counts


registry = {
    'nsynth_tfrecord': NSynthTFRecordDataset,
}
