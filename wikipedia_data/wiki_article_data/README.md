# Wiki Article Data

The actual wikipedia dump!


## Download

Torrent it: https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia

Also, get the index file here: https://dumps.wikimedia.org/enwiki/20210220/

Unzip it: `bunzip2 enwiki-20210220-pages-articles-multistream-index.txt.bz2`


## Facts

Size: 70-something gigs

Total number of articles: 20128863
`cat enwiki-20200401-pages-articles-multistream.xml | grep -i "<title>" | wc -l`

Confirmed to be 20993369
`wc -l enwiki-20210220-pages-articles-multistream-index.txt`

Total number of redirect articles: 9468507
`cat enwiki-20200401-pages-articles-multistream.xml | grep -i "<redirect title=" | wc -l`


https://en.wikipedia.org/wiki/Wikipedia#:~:text=The%20English%20Wikipedia%2C%20with%206.3,billion%20unique%20visitors%20per%20month