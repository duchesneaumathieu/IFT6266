#import alsaaudio
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from fuel.datasets.youtube_audio import YouTubeAudio

def get_sound(name):
    data = YouTubeAudio(name)
    stream = data.get_example_stream()
    it = stream.get_epoch_iterator()
    return next(it)[0][:, 0]

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
        
def playseq(seq, rate=16000, time=None):
    seq = np.asarray(seq, dtype=np.int16)
    if time is None: time=(0, seq.shape[0])
    seq = seq[time[0]*rate:time[1]*rate]
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
    it = time*16000
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
        f[i] = rfft(zero_padding(sound, start, end))
    f = np.asarray(f)
    return np.absolute(f), np.angle(f)

def FourierDecoder(amp, phase, batch_size, step_size=None):
    if step_size is None: step_size = batch_size
    fourier_seq = np.multiply(amp, np.exp(np.complex(0,1)*phase))
    f = np.zeros((len(fourier_seq)-1)*step_size+batch_size)
    n = np.zeros(len(f))
    for i in range(len(fourier_seq)):
        start = i*step_size
        end = start+batch_size
        f[start:end] += irfft(fourier_seq[i])
        n[start:end] += np.ones(batch_size)
    return np.asarray(np.real(f/n), dtype=np.int16)

def spectro(amp, hz=16000, fstep=500, max_sec=30, figsize=None, cmap='inferno', time=None):
    if figsize is None: figsize=(40, hz/16000.0*25)
    h = amp.shape[1]
    if time is not None: amp = amp[(time[0]*hz)/(2*h-2):(time[1]*hz)/(2*h-2)]
    w = amp.shape[0]
    leap_sec = (w*(2*h-2)/hz)/max_sec + 1
    step = (leap_sec*hz)/float(2*h-2)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(np.sqrt(amp.T), cmap=cmap, aspect='auto')
    plt.xticks(np.arange(0, w, step, dtype=np.int), np.arange(0, leap_sec*w/step, leap_sec, dtype=np.int))
    plt.yticks(np.arange(0, h, h/(hz/2/fstep)), np.arange(0,hz/2,fstep))
    plt.xlabel('t (s)')
    plt.ylabel('freq')
    plt.title('Spectrogram')
    
def boring_sound(time, freq, hz=16000, amp=32000):
    return amp*np.asarray([np.sin(2*np.pi*freq*i/hz) for i in range(int(time*hz))])

def less_boring_sound(time, freqs, hz=16000, amp=32000):
    ans = []
    for f in freqs: ans+=boring_sound(time, f, hz=hz, amp=amp).tolist()
    return np.asarray(ans)
