import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rates, samples = wavfile.read("C:/Users/Tim/Music/NotesTrainingData/A1-0.wav")
frequencies, time, spectrogram = signal.spectrogram(samples, sample_rates)
plt.imshow(spectrogram)
# plt.pcolormesh(times, frequencies, spectrogram)
plt.show()