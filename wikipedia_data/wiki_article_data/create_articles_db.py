import re
import json
from lxml import etree

a = r"(<[^>]+>|\[\d+\]|[,.\'\"()])"
ALL_DATA = "enwiki-20200401-pages-articles-multistream.xml"
SMALL_DATA = "enwiki.small.xml"


import untangle

doc = untangle.parse(SMALL_DATA)
for page in doc.mediawiki.page:
    print(page.title.cdata)
    print(page.id.cdata)
    print(page["redirect"])
    print(page.revision.text.cdata.replace("\n", "")[:500])


    # for text in page.revision.text:
    #     t = text.cdata.replace("\n", "")
    #     print(t[:50])
    print("---------")