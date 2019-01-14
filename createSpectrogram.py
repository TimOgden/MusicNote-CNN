import pyaudio
import wave, struct
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

p = pyaudio.PyAudio()
rate = 44100
chunk = 1024
recording_length = .5
channels = 2
p_format = pyaudio.paInt16
threshold_silence = 0
stream = p.open(format=p_format,
				channels=channels,
				rate=rate,
				input=True,
				frames_per_buffer=chunk)



def record_for_time(time):
	print('* recording')

	frames = []

	for i in range(int(rate/chunk*5)):
		data = stream.read(chunk)
		frames.append(data)

	print('* done recording')

	stream.stop_stream()
	stream.close()
	p.terminate()

	with wave.open('output.wav', 'wb') as f:
		f.setnchannels(channels)
		f.setsampwidth(p.get_sample_size(p_format))
		f.setframerate(rate)
		f.writeframes(b''.join(frames))
	with wave.open('output.wav','r') as f:
		plot_wav(f)
		find_whitespace(f, frames)
		print(start)

def plot_wav(file):
	signal = file.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	plt.figure(1)
	plt.plot(signal)
	plt.show()

def find_whitespace(file, frames):
	length = file.getnframes()
	for i in range(0,length):
	    waveData = file.readframes(1)
	    data = struct.unpack("<h", waveData)
	    print(int(data[0]))
	'''
	signal = file.readframes(-1)
	print(len(frames))
	signal = np.fromstring(signal, 'Int16')
	start = 0
	print(signal)
	plt.figure(1)
	plt.plot(signal)
	plt.show()
	
	for c, val in enumerate(signal):
		if abs(val) >= threshold_silence:
			start = c
		else:
			print(abs(val))
	'''

record_for_time(recording_length)