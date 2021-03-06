try:
  import librosa
except ImportError:
  pass
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read as spwavread, write as spwavwrite


def decode_audio(fp, fs=None, mono=False, normalize=False, fastwav=False):
  """Decodes audio file paths into 32-bit floating point vectors.

  Args:
    fp: Audio file path.
    fs: If specified, resamples decoded audio to this rate.
    mono: If true, averages channels to mono.
    fastwav: Assume fp is a standard WAV file (PCM 16-bit or float 32-bit).

  Returns:
    A np.float32 array containing the audio samples at specified sample rate.
  """
  if fastwav:
    # Read with scipy wavread (fast but only supports standard WAV files).
    try:
      orig_fs, x = spwavread(fp)
    except:
      raise ValueError('Error encountered when decoding WAV file.')
    if fs is not None and fs != orig_fs:
      raise ValueError('Fastwav cannot resample audio.')
    fs = orig_fs
    if x.dtype == np.int16:
      x = x.astype(np.float32)
      x /= 32768.
    elif x.dtype == np.float32:
      pass
    else:
      raise ValueError('Fastwav cannot process atypical WAV files.')
  else:
    # Decode with librosa load (slow but supports more file formats).
    if not librosa:
      raise Exception('Please install librosa')
    try:
      x, fs = librosa.core.load(fp, sr=fs, mono=False)
    except:
      raise ValueError('Error encountered when decoding audio file.')
    if x.ndim == 2:
      x = np.swapaxes(x, 0, 1)

  assert x.dtype == np.float32

  # At this point, x is np.float32 either [nsamps,] or [nsamps, nch].
  # We want [nsamps, 1, nch] to mimic 2D shape of spectral feats.
  if x.ndim == 1:
    nsamps = x.shape[0]
    nch = 1
  else:
    nsamps, nch = x.shape
  x = np.reshape(x, [nsamps, 1, nch])
 
  # Average channels if we want monaural audio.
  if mono:
    x = np.mean(x, 2, keepdims=True)

  if normalize:
    factor = np.max(np.abs(x))
    if factor > 0:
      x /= factor

  return fs, x

# TODO: Add dynamic typing
def audio_preprocess_tf(x, conversion_needed=True):
  print("Here is the shape of waveform before preprocess")
  print(x.get_shape().as_list())
  if len(x.get_shape().as_list()) == 2:
    batches = x.shape[0]
    nsamps = x.shape[1]
    nch = 1
  else:
    batches, nsamps, nch = x.shape
  x = tf.reshape(x, [batches, nsamps, 1, nch])
  if conversion_needed:
    x /= 32768.
  return x

def audio_postprocess_tf(x, conversion_needed=True):
  print("Here is the shape of waveform before postprocess")
  print(x.get_shape().as_list())
  x = x[:, :, 0, 0]
  if conversion_needed:
    x *= 32768.
    x = tf.clip_by_value(x, -32768., 32767.)
  return x


def save_as_wav(fp, fs, x):
  """Saves floating point waveform as signed 16-bit PCM WAV file.

  Args:
    fp: Output file path.
    fs: Waveform sample rate.
    x: Waveform (must be 32-bit float nd-array of size [?, 1, 1])
  """
  print("Saving audios")
  print(x.shape)
  try:
    nsamps, nfeats, nch = x.shape
  except ValueError:
    raise ValueError('Incorrect number of input dimesions.')
  if nfeats != 1:
    raise ValueError('Incorrect input dimesions.')
  if nch != 1:
    raise NotImplementedError('Can only save monaural WAV for now.')

  x = np.copy(x[:, 0, 0])

  x *= 32768.
  x = np.clip(x, -32768., 32767.)
  x = x.astype(np.int16)
  spwavwrite(fp, fs, x)