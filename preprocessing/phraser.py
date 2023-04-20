from gensim.models.phrases import Phrases, Phraser
from nltk import word_tokenize
import os


def get_sentences_from_file(path):
    with open(path) as fin:
        sentences = fin.readlines()

    return [word_tokenize(s) for s in sentences]


# Given sentences as lists of tokens, run phraser to detect bigrams
# Input can be a list of sentences (single corpus or document)
#       or a list of documents (as a list of tokenized sentences)
# Output is a list of sentences with tokenize bigrams
# Bigrams are connected with delim
def train_phrase_model(sentences=None, documents=None, threshold=0.5,
                       save=False, delim=b'_'):
    if not sentences and not documents:
        print("(!) Error: Phraser input not provided!")
        return None

    if documents:
        # If documents were provided, train the phraser on the entire data
        # before applying the ngrams
        sentences = [sent for doc in documents for sent in doc]

    phrases = Phrases(sentences, min_count=5, threshold=threshold,
                      delimiter=delim, scoring='npmi')
    ngram = Phraser(phrases)

    if save:
        phrases.save("phrases.model")  # save phrases (re-trainable) model
        ngram.save("phraser.model")  # save fast non-trainable model

    if documents:
        # Create new documents by applying ngrams to its sentences
        documents = [[ngram[sent] for sent in doc] for doc in documents]
        # Since Phraser does not accept empty delimiters, we will remove them
        # manually here by replacing - with ''
        # Using delim b'-' as the NULL delimiter
        if delim == b'-':
            # Comprehension to replace "-" in every string of the document
            documents = [[[w.replace("-", "") for w in sent] for sent in doc]
                         for doc in documents]

        return documents
    else:
        sentences = ngram[sentences]
        # Since Phraser does not accept empty delimiters, we will remove them
        # manually here by replacing - with ''
        # Using delim b'-' as the NULL delimiter
        if delim == b'-':
            new_sentences = list()
            for sent in sentences:
                new_sent = [w.replace("-", "") for w in sent]
                new_sentences.append(new_sent)
            sentences = new_sentences
        return sentences


def main():
    data_path = "/home/gouvem/bias/data/"

    sentences = list()
    for root, dir, files in os.walk(data_path):
        for f in files:
            print(f)
            sentences.extend(get_sentences_from_file(os.path.join(root, f)))

    train_phrase_model(sentences)


if __name__ == "__main__":
    main()
