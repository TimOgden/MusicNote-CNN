import pyaudio
import wave, struct
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import random
from searchChord import search_google
import sys, os


p = pyaudio.PyAudio()
rate = 44100
chunk = 1024
recording_length = 4
channels = 1
p_format = pyaudio.paInt16

#These were calculated using my voice, may need to change with guitar
start_threshold_silence = 30
end_threshold_silence = 30

stream = p.open(format=p_format,
				channels=channels,
				rate=rate,
				input=True,
				frames_per_buffer=chunk)



def record_for_time(time, filename):
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
						'chord_type': ['major', 'minor', 'diminished', 'major 7th', 'minor 7th', 'dominant 7th',
									 'sus2', 'sus4', 'augmented', 'extended', 'octave', '5th']} 
									 # The octave and 5th are intervals, not chords, but they are so fundamental to rock music that
									 	# I figured I must include them.

def gen_random_chord():
	note = random.randrange(0,len(chord_customizers['root_note']))
	c_type = random.randrange(0,len(chord_customizers['chord_type']))
	return chord_customizers['root_note'][note] + " " + chord_customizers['chord_type'][c_type]

def plot_chords(chordDir):
	path="C:\\Users\\Tim\\ProgrammingProjects\\MusicNote-CNN\\recordings\\downloads\\" + chordDir + " guitar chord"
	num_rows = 3
	num_cols = 3
	num_images = 3*3

	plt.figure(figsize=(2*2*num_cols, num_rows))
	plt.title(chordDir)
	i = 0
	for filename in os.listdir(path):
		plt.subplot(num_rows, 3*num_cols, 3*i+1)
		plot_image(path, filename)
		i+=1
	plt.show()

def plot_image(path, img):
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	img = mpimg.imread(path + "\\" + img)
	plt.imshow(img, cmap=plt.cm.binary)

each_chord = {}


#record_for_time(recording_length, 'output.wav')


while True:
	chord = gen_random_chord()
	search_google(chord)
	plot_chords(chord)
	print(chord)

	if chord in each_chord:
		each_chord[chord] += 1
	else:
		each_chord[chord] = 1
	
	file = ""
	num_rep = 5
	for i in range(num_rep):
		print('taking step {} of {}'.format(i+1, num_rep))
		keep = 'n'
		while keep == 'n':
			file = chord + "-" + str(each_chord[chord]) + '.wav'
			record_for_time(recording_length, file)
			keep = input('Keep this recording? - ')
		if keep == 'q':
			os.remove(file)
			sys.exit(0)
		if keep == 'n':
			os.remove(file)


stream.stop_stream()
stream.close()
p.terminate()
