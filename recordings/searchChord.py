from google_images_download import google_images_download
import os



def search(val):
    arguments = {'keywords': val, 'limit': 9}
    #arguments = {'keywords': "banana", "limit": 5}
    response = google_images_download.googleimagesdownload()
    paths = response.download(arguments)

def search_google(chord):
    val = chord + " ukulele chord"
    if val not in os.listdir("C:\\Users\\Tim\\ProgrammingProjects\\MusicNote-CNN\\recordings\\downloads\\"):
        search(val)
