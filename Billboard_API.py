#Billboard Top 100 API
import billboard
import csv

songs = [['Rank', 'Title', 'Artist', 'Year']]
years = [2016, 2020]
for k in range(len(years)):
    chart = billboard.ChartData('hot-100-songs', year=years[k])
    print("this is a really long message to stand out and you will see this")
    for i in range(99):
        song = chart[i]  # Get no. 1 song on chart
        title = song.title
        artist = song.artist
        genre = song.genre
        print(genre)
        print(artist)
        print(title)
        song = [i+1, artist, title, years[k]]
        songs.append(song)
with open("top_100_songs.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(songs)

