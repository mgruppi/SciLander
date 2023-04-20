from preprocessing.phraser import train_phrase_model
import re
import argparse
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Tokenize words in text to separate punctuation of words
def get_tokens(text, d_regex=None, sub_regex=None, del_regex=None,
               to_remove={}, min_size=3, max_size=40):

    # Combined hyphenated words into a single term
    text = re.sub("[-,.\_]", "", text)

    # Remove URLS
    # Replace any occurrence of http(s)://<anything>
    # Until a space or end-of-string is reached
    # Since some URLs can contain \n in them which are obtained during the
    # scraping, the replacement will include URLs broken by \n
    text = re.sub("http(s?)://.+(\s|$)", " ", text)

    # Use regex split which removes all non-word characters (including punkt)
    # Discard tokens that outside bounds for (min_size, max_size)
    # Also discard words in to_remove set (typically stopwords)
    tokens = [t.lower() for t in re.split("\W+", text)
              if min_size <= len(t) <= max_size and t.lower() not in to_remove]

    return tokens


# Do a first pass over the text to remove noisy and repeat urls
# (since word_tokenize does not properly handle thoese)
# Matches any string surround by space or newline (and start/end of string)
def split_tokens(text, min_length=0, max_length=30):
    r = re.compile("(\s|\n)")
    splits = r.split(text)

    return "".join(s for s in splits if min_length <= len(s) <= max_length)


def lemmatize_sentence(sentence):
    """
    Lemmatize tokens with NLTK
    Returns a list of lemmatized tokens (sentence)
    """
    lemmatizer = WordNetLemmatizer()

    lemm_sent = list()

    for t in sentence:
        lemm_sent.append(lemmatizer.lemmatize(t))
    return lemm_sent


# Convert text to LineSentence format (one sentence per line)
def get_tokenized_sentences(text, stopword_lang="english"):
    # Given cursor to database result fetching title, cotent,
    # Return list of sentences in LineSentence format
    # (https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence)
    sentences = list()
    # Get set of nltk's stopwords
    stop_words = set(stopwords.words(stopword_lang))
    # Get nltk's sentence detector
    sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    content_sents = sent_detector.tokenize(text)
    # Tokenize each sentence individually
    for sent in content_sents:
        tks = get_tokens(sent, to_remove=stop_words)
        tks = lemmatize_sentence(tks)  # lemmatize
        if len(tks) > 0:
            sentences.append(tks)
    return sentences


# Pre-process input text
# Input: text as string or documents as list of strings
#           - use_phrases toggles on/off the bigram detector
#           - bigram_delim is the character use to connect bigrams
# Output: list of lists of tokens (sentences)
#           or documents as list of lists of lists of tokens
def preprocess_text(text=None, documents=None,
                    use_phrases=True, bigram_delim=b'_'):
    if not text and not documents:
        print("(!) Error: text or document input not provided")
        return None
    print("\t+ Tokenizing")

    if text:
        sentences = get_tokenized_sentences(text)
        documents = None
    elif documents:
        new_docs = list()
        for doc in documents:
            new_docs.append(get_tokenized_sentences(doc))
        documents = new_docs
        sentences = None

    if use_phrases:
        print("\t+ Phrasing...")
        sentences = train_phrase_model(sentences=sentences,
                                       documents=documents, delim=bigram_delim)

    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input text file")
    parser.add_argument("output", type=str, help="Output processed text file")
    parser.add_argument("--no_phrases", action="store_const", default=False,
                        const=True,
                        help="Turn off phrases (bigrams) to output")

    args = parser.parse_args()

    print("Input", args.input)
    with open(args.input) as fin:
        text = "".join(line for line in fin)

    sentences = preprocess_text(text, use_phrases=not args.no_phrases)

    with open(args.output, "w") as fout:
        for sent in sentences:
            fout.write(" ".join(w for w in sent))
            fout.write("\n")


if __name__ == "__main__":
    main()
