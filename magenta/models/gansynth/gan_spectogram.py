from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import re

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

import absl.flags
from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import generate_util as gu
from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib import util
import tensorflow as tf

absl.flags.DEFINE_string('ckpt_dir',
                         '/tmp/gansynth/acoustic_only',
                         'Path to the base directory of pretrained checkpoints.'
                         'The base directory should contain many '
                         '"stage_000*" subdirectories.')
absl.flags.DEFINE_string('output_dir',
                         '/tmp/gansynth/samples',
                         'Path to directory to save wave files.')
absl.flags.DEFINE_integer('batch_size', 8, 'Batch size for generation.')
absl.flags.DEFINE_string('pitches', '36', 'Note pitches to generate.')
absl.flags.DEFINE_integer('resolution', 9, 'Resolution of interpolation grid.')

FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def make_grid(res):
  x, y = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res))
  x = x.reshape(-1)
  y = y.reshape(-1)
  return zip(x, y)

def get_weights(xy):
	corners = np.array([[0,0], [1,0], [0,1], [1,1]])
	distances = np.linalg.norm(xy - corners, axis=1)
	distances = np.maximum(1 - distances, 0)
	distances /= distances.sum()
	return distances

def note_meta(combination, weights):
	meta = {}
	for i, w in zip(combination, weights):
		if w > 0:
			meta[i] = np.around(w, decimals=3)
	return hashabledict(meta)

def meta_to_name(meta):
	attr = []
	for key in sorted(meta.keys()):
		attr.append(key)
		attr.append(meta[key])
	return '_'.join(map(str, attr))

def get_z_notes(z_instruments, instrument_names, xy_grid):
  z_notes = []
  metas = []
  for i, (x, y) in enumerate(xy_grid):
    weights = get_weights((x, y))

    z_interpolated_01 = gu.slerp(z_instruments[0], z_instruments[1], x)
    z_interpolated_23 = gu.slerp(z_instruments[2], z_instruments[3], x)
    z_interpolated = gu.slerp(z_interpolated_01, z_interpolated_23, y)
    #z_interpolated = (z_instruments.T * weights).T.sum(axis=0)
    z_notes.append(z_interpolated)

    metas.append(note_meta(instrument_names, weights))

  return np.vstack(z_notes), metas

def gen_instrument_name(n):
  cs = "bcdfghjklmnpqrstvwxz"
  vs = "aeiouy"
  v = random.random() < 0.33;
  name = ""
  for i in range(n):
    name += random.choice(vs if v else cs)
    v = not v
    
  return name

def parse_pitches(text):
  pat = r"^(?:\((-?\d+)(?:,(-?\d+))?\.\.(-?\d+)\)|(-?\d+))(?:|,(.*))$"
  rem = text
  pitches = []
  while rem:
    m = re.match(pat, rem)
    if not m:
      raise Exception("invalid format in pitches: {}".format(text))
    
    int_if_some = lambda x: int(x) if x != None else None
    range_0 = int_if_some(m.group(1))
    range_1 = int_if_some(m.group(2))
    range_max = int_if_some(m.group(3))
    single = int_if_some(m.group(4))
    rem = m.group(5)
    
    if single != None:
      pitches.append(single)
      continue
    
    if range_0 != None and range_max != None:
      step = 1
      if range_1 != None:
        step = range_1 - range_0
      pitches.extend(range(range_0, range_max+1, step))
      continue
  
  return pitches

def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True

  # Load the model
  flags = lib_flags.Flags({'batch_size_schedule': [FLAGS.batch_size]})
  model = lib_model.Model.load_from_path(FLAGS.ckpt_dir, flags)

  # Make an output directory if it doesn't exist
  output_dir = util.expand_path(FLAGS.output_dir)

  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  # generate 4 random latent vectors
  z_instruments = model.generate_z(4)
  instrument_names = list(gen_instrument_name(random.randint(3, 8)) for _ in range(4))
  
  # interpolate
  res = FLAGS.resolution
  pitches = parse_pitches(FLAGS.pitches)
  xy_grid = make_grid(res)

  print()
  print("resolution =", res)
  print("pitches =", pitches)
  print("z_instruments.shape =", z_instruments.shape)
  print("z_instruments =", z_instruments)
  print("instrument_names =", instrument_names)
  
  z_notes, note_metas = get_z_notes(z_instruments, instrument_names, xy_grid)
  print("z_notes.shape =", z_notes.shape)

  z_notes_rep = np.repeat(z_notes, len(pitches), axis=0)
  print("z_notes_rep.shape =", z_notes_rep.shape)

  pitches_rep = pitches * z_notes.shape[0]
  print("len(pitches_rep) =", len(pitches_rep))
  
  print("generating {} samples,,".format(len(z_notes_rep)))
  audio_notes = model.generate_samples_from_z(z_notes_rep, pitches_rep)
  
  audio_metas = []
  for note_meta in note_metas:
    for pitch in pitches:
      meta = dict(note_meta)
      meta["pitch"] = pitch
      audio_metas.append(meta)
  
  print("audio_notes.shape =", audio_notes.shape) 
  print("len(audio_metas) =", len(audio_metas))
  
  fig, axs = plt.subplots(res, res)
  fig.suptitle('Latent Space')

  for i, (wave, meta) in enumerate(zip(audio_notes, audio_metas)):
    name = meta_to_name(meta)
    fn = os.path.join(output_dir, "gen_{}.wav".format(name))
    x = i % res
    y = int(math.floor(i / res))
    axs[x, y].specgram(wave,Fs=16000) 
    axs[x, y].get_xaxis().set_visible(False)
    axs[x, y].get_yaxis().set_visible(False)

    
  plt.show()
    
def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
