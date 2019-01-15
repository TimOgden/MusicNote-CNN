from google_images_download import google_images_download




def search(val):
    arguments = {'keywords': val, 'limit': 9}
    #arguments = {'keywords': "banana", "limit": 5}
    response = google_images_download.googleimagesdownload()
    paths = response.download(arguments)
    print(paths)

def search_google(chord):
    search(chord + " guitar chord")
