import PIL
import glob, os
from PIL import Image

os.chdir("/TrainingData")
for file in glob.glob("*.jpeg"):
	file.resize((115,115))