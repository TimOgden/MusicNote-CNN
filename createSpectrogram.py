import pyaudio
import wave, struct
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment


p = pyaudio.PyAudio()
rate = 44100
chunk = 1024
recording_length = 5
channels = 1
p_format = pyaudio.paInt16
start_threshold_silence = 50
end_threshold_silence = 70
stream = p.open(format=p_format,
				channels=channels,
				rate=rate,
				input=True,
				frames_per_buffer=chunk)



def record_for_time(time, filename):
	print('* recording')

	frames = []

	for i in range(int(rate/chunk*time)):
		data = stream.read(chunk)
		frames.append(data)

	print('* done recording')

	stream.stop_stream()
	stream.close()
	p.terminate()


	start = 0
	end = 0
	with wave.open(filename, 'wb') as f:
		f.setnchannels(channels)
		f.setsampwidth(p.get_sample_size(p_format))
		f.setframerate(rate)
		f.writeframes(b''.join(frames))
	with wave.open(filename,'r') as f:
		start, end = find_whitespace(f, frames)
		plot_wav(f, dispEnds=True, start=start, end=end)
		#print('Start: {}, End: {}'.format(start,end))

	start_millis = int(start * recording_length * 1000) # Convert to millis
	end_millis = int(end * recording_length * 1000)
	newAudio = AudioSegment.from_wav(filename)
	newAudio = newAudio[start_millis:end_millis]
	newAudio.export(filename, format="wav")

	with wave.open(filename, 'r') as f:
		plot_wav(f)

def plot_wav(file, dispEnds=False, start=0, end=0):
	signal = file.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	plt.figure(1)
	plt.plot(signal)
	if dispEnds:
		plt.axvline(start, color='r')
		plt.axvline(end, color='r')
	plt.show()

def find_whitespace(file, frames):
	[fs, x] = wavfile.read('output.wav')
	start = 0
	for c, val in enumerate(x):
		if abs(val) >= start_threshold_silence:
			start = c
			break
	

	end = len(x) - 1
	#print('Len:', end)
	for val in range(len(x)-1, -1, -1):
		if abs(x[val]) >= end_threshold_silence:
			#print('val', val)
			break
		end -= 1
	return start/len(x), end/len(x)

record_for_time(recording_length, 'output.wav')