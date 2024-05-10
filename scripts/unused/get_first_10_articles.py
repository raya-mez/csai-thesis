import gzip
import json

input_file_path = 'data/enwiki-latest.json.gz'
output_file_path = 'data/enwiki-first10articles.json'

# Initialize a list to hold the first 10 articles
articles = []

# Open the compressed JSON file and read the first 10 items
with gzip.open(input_file_path, 'rt', encoding='utf-8') as f:
    for i in range(10): 
        line = f.readline()
        if not line:  # Break if the file ends before reaching 10 items
            break
        article = json.loads(line)
        articles.append(article)

# Write the articles to an uncompressed JSON file
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(articles, outfile, indent=4)
