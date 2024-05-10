from gensim import utils
import json

# iterate over the plain text data we just created
with utils.open('data/enwiki-latest.json.gz', 'rb') as f:
    for line in f:
        # decode each JSON line into a Python dictionary object
        article = json.loads(line)
        # each article has a "title", a mapping of interlinks and a list of "section_titles" and
        # "section_texts".
        print("Article title: %s" % article['title'])
        print("Interlinks: %s" + article['interlinks'])
        for section_title, section_text in zip(article['section_titles'], article['section_texts']):
            print("Section title: %s" % section_title)
            print("Section text: %s" % section_text)