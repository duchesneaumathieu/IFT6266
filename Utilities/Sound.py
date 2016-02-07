import alsaaudio
import numpy as np
from numpy.fft import fft, ifft

def play(file_name):

    f = open(file_name, 'rb')

    # Open the device in playback mode. 
    out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK)

    # Set attributes: Mono, 44100 Hz, 16 bit little endian frames
    out.setchannels(1)
    out.setrate(16000)
    out.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.
    out.setperiodsize(512) #160

    # Read data from stdin
    data = f.read(512) #320
    while data:
        out.write(data)
        data = f.read(512)
        
def playseq(seq, rate=16000):
    seq = np.asarray(seq, dtype=np.int16)
    # Open the device in playback mode. 
    out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK)

    # Set attributes: Mono, 44100 Hz, 16 bit little endian frames
    out.setchannels(1)
    out.setrate(rate)
    out.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.
    out.setperiodsize(512) #160

    step = 512 #320
    i = 0
    j = min(i+step, seq.shape[0])
    while i < seq.shape[0]:
        data = seq[i: j]
        out.write(data)
        i = j
        j = min(i+step, seq.shape[0])
        
def playfreq(freq, time=3):
    it = time*15000
    sound = ifft(freq)
    sound = np.asarray(sound.tolist()*int(it/len(sound)))
    playseq(sound)

def zero_padding(t, i, j):
    if i >= len(t): return np.asarray([0.]*(j-i))
    if j >= len(t): return np.append(t[i:len(t)], [0.]*(j-len(t)))
    return t[i:j]

def FourierEncoder(sound, batch_size, step_size=None):
    if step_size is None: step_size = batch_size
    f = [0]*int(np.ceil(len(sound)/(1.*step_size)))
    for i in range(len(f)):
        start = i*step_size
        end = start+batch_size
        f[i] = fft(zero_padding(sound, start, end))
    return np.asarray(f)

def FourierDecoder(fourier_seq, batch_size, step_size=None):
    if step_size is None: step_size = batch_size
    f = np.asarray([0]*((len(fourier_seq)-1)*step_size+batch_size), dtype=np.complex128)
    n = np.asarray([0.]*len(f))
    for i in range(len(fourier_seq)):
        start = i*step_size
        end = start+batch_size
        f[start:end] += ifft(fourier_seq[i])
        n[start:end] += np.ones(batch_size)
    return f/n