# Meeting 10/22/19
# TODOs
- something more useful than n-grams
- look more into "Styles"


# Next steps
## Toxicity
- keep perspective running
- can we link it back to a site?
    - what is the overlap of toxicity scores with 4chan links?
- manually label some data (with the reading group?)
- enumerate types of toxic language
    - hate towards undocumented immigrants
    - anti-semetic speech
    - islamaphobia
    /START WITH THIS:/
    - 4chan -- can we scrape or is there a corpus?
    - breitbart
- use vampire to see if we can find gold-labeled 4chan data in the openwebtext dataset

## Styles
- think more about types of styles of text
- create gold-labeled text of that style
    - ngrams, tf-idf, bert
    - use to cross-reference the training data
- find nearest neighbors
    - https://github.com/facebookresearch/faiss
- find statistics about occurrence
- given a context relating to, eg. 4chan, establish that gpt-2 generates these comments


# Resources
## Software
scikit-learn count vectorizer
gnu parallel
siteresolver

## Papers



---


