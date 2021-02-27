"""
This creates the database wiki_article_metadata.db. The URI was chosen as the primary key because it should
be unique. This primary key will be joined to data extracted from the full wikipedia dump.
"""
import re
import sqlite3


# Connect to the database.
conn = sqlite3.connect("wiki_article_metadata.db")
c = conn.cursor()

# Create a table.
c.execute("""CREATE TABLE wiki_metadata
             (uri text primary key, title text, pagerank real, pageviews real)""")

# Iterate through the wiki metadata files line by line and add them to the database.
path = "."

for uri, title, pagerank, pageview in zip(
    open(f"{path}/enwiki-2020.uris"),
    open(f"{path}/enwiki-2020.titles"),
    open(f"{path}/enwiki-2020-pr-3.txt"),
    open(f"{path}/enwiki-2020-pv.txt")):

    # Basic sanitizing: remove all newlines
    title = re.sub("\n", "", title)
    uri = re.sub("\n", "", uri)

    # Double up single quotes so they're escaped when inserting
    title = re.sub("\'", "\'\'", title)
    uri = re.sub("\'", "\'\'", uri)

    c.execute(f"""INSERT INTO wiki_metadata VALUES ('{uri}', '{title}', {pagerank}, {pageview})""")

conn.commit()
conn.close()
