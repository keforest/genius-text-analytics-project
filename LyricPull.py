import billboard
import csv
import lyricsgenius as lg
from csv import reader

file = open("Lyrics_2020", "w")  # File to write lyrics to
genius = lg.Genius('emlEeWcpKcHjCQmlxb2jkJZbGrKnVaky55igLwmeOw-PNC8b0r6A3tL7W-asCW7J',  # Client access token from Genius Client API page
                             skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"],
                             remove_section_headers=True)

#Set 100 top song csv as a list of lists
with open('top_100_songs_2020.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    list_of_rows = list(csv_reader)

#Pull from genius API
def get_lyrics(arr):  # Write lyrics of k songs by each artist in arr
    c = 0  # Counter
    try:
        song = genius.search_song(arr[1], arr[0])
        s = song.lyrics
        c += 1
        print(f"Song grabbed")
    except:  #  Broad catch which will give us the name of artist and song that threw the exception
        print(f"some exception at {name}: {c}")
    file.write(s)

#Change One Dance to just being written by Drake
data = [['Rank', 'Title', 'Artist', 'Year', 'Lyrics']]
#Pull lyrics for each song
for k in range(len(list_of_rows)):
    artist_song = [list_of_rows[k][1], list_of_rows[k][2]]
    get_lyrics(artist_song)
    data = [list_of_rows[k][0], list_of_rows[k][1], list_of_rows[k][2], list_of_rows[k][3]]

#A really cool sub project could be to seperate documents by genre and see if the topic clustering can actually determine
#Genre based on solely lyrics

