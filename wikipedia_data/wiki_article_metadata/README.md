# Wikipedia relevancy metadata

Information about article pageviews, pagerankings, etc. is from here: http://wikirank-2020.di.unimi.it/faq.html

This information is meant to act as a filter when performing document retrieval - only consider documents with high pagerank scores/pageview numbers instead of looking at every document.


## Get the data

Pagerank
`curl http://wikirank-2020.di.unimi.it/enwiki-2020-pr-3.txt -O`

Page views
`curl http://wikirank-2020.di.unimi.it/enwiki-2020-pv.txt -O`

Titles
`curl http://wikirank-2020.di.unimi.it/enwiki-2020.titles -O`

URI's
`curl http://wikirank-2020.di.unimi.it/enwiki-2020.uris -O`

This is a __lot__ of data: `wc -l enwiki-2020.titles` is 6,047,510!


## Get it into a database

_Note:_ This takes 5-10 minutes to run:

`python3 create_db.py`

This creates the file `wiki_article_metadata.db` that can be inspected with `sqlite3`.
