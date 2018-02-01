__author__ = 'val'

from bs4 import BeautifulSoup
import urllib
import re
import os

dict = set()

def download_midi_recursive(website, page, folder):
    if page in dict:
        return

    dict.add(page)
    print("Downloading page " + page)

    html_page = urllib.request.urlopen(website + '/' + page)
    soup = BeautifulSoup(html_page, "lxml")
    for link in soup.findAll('a'):
        url = '{}'.format(link.get('href'))

        if url.endswith('.mid'):
            try:
                filename = os.path.basename(url)
                midiurl = urllib.request.urlopen(website + '/' + url)
                fullpath = folder + '/' + filename

                if os.path.exists(fullpath):
                    print("Skipping " + filename)
                else:
                    print("Downloading " +  filename)
                    with open(fullpath, "wb") as local_file:
                        content = midiurl.read()
                        local_file.write(content)

            except urllib.request.HTTPError as e:
                print("Http error" + e.code + url)
            except urllib.request.URLError as e:
                print("Url error" + e.reason + url)
        if url.endswith('.htm') or url.endswith('.asp'):
            try:
                relativeurl = os.path.basename(url)
                download_midi_recursive(website, relativeurl, folder)
            except Exception as e:
                print(str(e))

# website = "http://www.midiworld.com"
# website = "http://www.piano-midi.de"
website = "http://www.piano-e-competition.com"
# page = "classic.htm"
# page = "midi_files.htm"
page = "midiinstructions.asp"
folder = './music_alt'
download_midi_recursive(website, page, folder)