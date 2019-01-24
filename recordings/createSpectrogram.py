import pyaudio
import wave, struct
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment
import random
from searchChord import search_google
import sys, os
import time


p = pyaudio.PyAudio()
rate = 44100
chunk = 1024
recording_length = 3
channels = 1
p_format = pyaudio.paInt16


start_threshold_silence = 120
end_threshold_silence = 30

chords_plot_time = 15
wav_plot_time = 1.5

stream = p.open(format=p_format,
				channels=channels,
				rate=rate,
				input=True,
				frames_per_buffer=chunk)

note_freq = {'A': [27.50,55.00,110.00,220.00,440.00,880.00,1760.00,3520.00,7040.00],
				'A#': [29.14,58.27,116.54,233.08,466.16,932.33,1864.66,3729.31,7458.62],
				'B': [30.87,61.74,123.47,246.94,493.88,987.77,1975.53,3951.07,7902.13],
				'C': [16.35,32.70,65.41,130.81,261.63,523.25,1046.50,2093.00,4186.01],
				'C#': [17.32,34.65,69.30,138.59,277.18,554.37,1108.73,2217.46,4434.92],
				'D': [18.35,36.71,73.42,146.83,293.66,587.33,1174.66,2349.32,4698.63],
				'D#': [19.45,38.89,77.78,155.56,311.13,622.25,1244.51,2489.02,4978.03],
				'E': [20.60,41.20,82.41,164.81,329.63,659.25,1318.51,2637.02,5274.04],
				'F': [21.83,43.65,87.31,174.61,349.23,698.46,1396.91,2793.83,5587.65],
				'F#': [23.12,46.25,92.50,185.00,369.99,739.99,1479.98,2959.96,5919.91],
				'G': [24.50,49.00,98.00,196.00,392.00,783.99,1567.98,3135.96,6271.93],
				'G#': [25.96,51.91,103.83,207.65,415.30,830.61,1661.22,3322.44,6644.88]}

colors = ['b','m','r','c','k','w']

def record_for_time(time, filename, notes=None, plot_spectrogram=True):
	#filename = "\\recordings\\" + filename
	print('* recording')

	frames = []

	for i in range(int(rate/chunk*time)):
		data = stream.read(chunk)
		frames.append(data)

	print('* done recording')

	start = 0
	end = 0
	start_sec = 0
	end_sec = 0
	with wave.open(filename, 'wb') as f:
		f.setnchannels(channels)
		f.setsampwidth(p.get_sample_size(p_format))
		f.setframerate(rate)
		f.writeframes(b''.join(frames))

	with wave.open(filename,'r') as f:
		start, end, start_sec, end_sec = find_whitespace(filename, frames)
		plot_wav(f, dispEnds=True, start=start, end=end)

	start_millis = int(start_sec * recording_length * 1000) # Convert to millis
	end_millis = int(end_sec * recording_length * 1000)
	newAudio = AudioSegment.from_wav(filename)
	newAudio = newAudio[start_millis:end_millis]
	newAudio.export(filename, format="wav")

	if plot_spectrogram:
		plot_spect(filename, notes, dispNotes=False)
	
	#plt.pause(wav_plot_time)
	#plt.close()
	# Plot again to see what was removed
	#with wave.open(filename, 'r') as f:
		#plot_wav(f)

def plot_spect(file, notes, dispNotes=False):
	freeze = False

	sample_rates, samples = wavfile.read(file)
	frequencies, times, spectrogram = signal.spectrogram(samples,sample_rates,nfft=2048, noverlap=1800, nperseg=2048)
	fig = plt.figure(frameon=False)
	try:
		plt.pcolormesh(times, frequencies, 10*np.log10(spectrogram))
	except:
		pass
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	
	if dispNotes:
		displayNotes(notes, 4, 8)
		
	plt.ylim(top=8000)
	fig.savefig('C:/Users/Tim/ProgrammingProjects/MusicNote-CNN/recordings/spectrograms/' + file, bbox_inches='tight', pad_inches=0)
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.legend()
	if freeze:
		plt.show()
	else:
		plt.show(block=False)
		plt.pause(3)
		plt.close()
	

def displayNotes(notes, start_octave, end_octave):
	c = 0
	looped = False
	for note in notes:
		if c>=len(colors)-1:
			c = 0
			looped = True
		for i in range(start_octave, end_octave):
			plt.axhline(note_freq[note][i], color=colors[c], linewidth=3)
		c+=1


def plot_both(file1, file2):
	sample_rates1, samples1 = wavfile.read(file1)
	plt.figure(1)
	plt.subplot(211)
	frequencies1, times1, spectrogram1 = signal.spectrogram(samples1,sample_rates1,nfft=2048, noverlap=1800, nperseg=2048)

	plt.pcolormesh(times1, frequencies1, 10*np.log10(spectrogram1))
	
	sample_rates2, samples2 = wavfile.read(file2)
	plt.subplot(212)
	frequencies2, times2, spectrogram2 = signal.spectrogram(samples2,sample_rates2,nfft=2048, noverlap=1800, nperseg=2048)
	try:
		plt.pcolormesh(times2, frequencies2, 10*np.log10(spectrogram2), cmap=plt.cm.binary)
	except:
		pass
	plt.xlabel('Time')
	plt.ylabel('Frequency (Hz)')

	plt.show()

def plot_wav(file, dispEnds=False, start=0, end=0):
	freeze = False

	signal = file.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	plt.figure()
	plt.plot(signal)
	if dispEnds:
		plt.axvline(start, color='r')
		plt.axvline(end, color='r')
	if freeze:
		plt.show()
	else:
		plt.show(block=False)
		plt.pause(wav_plot_time)
		plt.close()
	

def find_whitespace(file, frames):
	[fs, x] = wavfile.read(file)
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
	return start, end, start/len(x), end/len(x)

chord_customizers = {'root_note': ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'],
						'chord_type': ['major', 'minor', 'major 7th', 'minor 7th',
									 'sus2', 'sus4', '8th interval', 'fifth interval']} 
									 # The octave and 5th are intervals, not chords, but they are so fundamental to rock music that
									 	# I figured I must include them.

def gen_random_chord():
	note = random.randrange(0,len(chord_customizers['root_note']))
	c_type = random.randrange(0,len(chord_customizers['chord_type']))
	return chord_customizers['root_note'][note] + chord_customizers['chord_type'][c_type]

def gen_random_note():
	note = random.randrange(0,len(chord_customizers['root_note']))
	return chord_customizers['root_note'][note]


def plot_chords(chordDir):
	path="C:\\Users\\Tim\\ProgrammingProjects\\MusicNote-CNN\\recordings\\downloads\\" + chordDir + " ukulele chord"
	num_rows = 3
	num_cols = 3
	num_images = 3*3

	plt.figure(chordDir, figsize=(2*2*num_cols, num_rows))
	plt.title(chordDir)
	i = 0
	for filename in os.listdir(path):
		plt.subplot(num_rows, 3*num_cols, 3*i+1)
		plot_image(path, filename)
		i+=1
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show(block=False)
	plt.pause(chords_plot_time)
	plt.close()

def plot_image(path, img):
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	try:
		img = mpimg.imread(path + "\\" + img)
		plt.imshow(img, cmap=plt.cm.binary)
	except:
		pass


each_chord = {}


#record_for_time(recording_length, 'output0.wav', plot_spectrogram=False)
#record_for_time(recording_length, 'output1.wav', plot_spectrogram=False)
#plot_both('output0.wav', 'output1.wav')


#record_for_time(recording_length, 'output0.wav', plot_spectrogram=True, notes=['C','G','E'])

c = 0
while True:
	#chord = gen_random_note()
	chord = gen_random_chord()
	search_google(chord)
	plot_chords(chord)
	print(chord)
	time.sleep(6)
	if chord in each_chord:
		each_chord[chord] += 1
	else:
		each_chord[chord] = 1
	
	if c % 5 == 0:
		print(each_chord)
	c+=1

	file = ""
	num_rep = 6
	for i in range(num_rep):
		print('Recording #{} of {}'.format(i+1, num_rep))
		keep = 'n'
		while keep == 'n':
			file = chord + "-" + str(each_chord[chord]) + '.png'
			time.sleep(1)
			record_for_time(recording_length, file, plot_spectrogram=True, notes=chord)
			keep = input('Keep this recording? - ')
		if keep == 'q':
			os.remove(file)
			sys.exit(0)
		if keep == 'n':
			os.remove(file)

		each_chord[chord] += 1


stream.stop_stream()
stream.close()
p.terminate()
