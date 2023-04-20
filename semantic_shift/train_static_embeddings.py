import sqlite3
import re
import os
import argparse
from mapping import read_vocabulary
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from gensim.models import Word2Vec
import spacy


def read_duplicates_file(path: str) -> set:
    """
    Returns the file in `path` and returns a list of article IDs found in it.
    The file in `path` must be a CSV file containing the pairs of article duplicates from a NELA database.
    Args:
        path: Path to the file.

    Returns:
        duplicates: Set of article ids found in the input file.

    """
    duplicates = set()

    with open(path) as fin:
        for line in fin:
            articles = line.strip().split(",", 1)
            for a in articles:
                duplicates.add(a)

    return duplicates


def get_patterns_and_substrings(vocab):
    """
    Given a vocabulary, constructs a list of patterns (regular expressions) and substitution strings.
    n-grams in the vocabulary are concatenated with _.
    The results can be used for n-gram substitution in the input documents.
    Args:
        vocab - list of vocabulary terms
    Returns:
        patterns, substrings - lists of patterns (regex) and substrings (str).
    """
    # Create regex from vocabulary
    patterns = list()
    sub_strings = list()
    for v in vocab:
        tokens = re.split("\W", v)
        r = re.compile(".".join(tokens))
        patterns.append(r)
        sub_strings.append("_".join(tokens))

    return patterns, sub_strings


def process_corpus_spacy(documents, nlp):
    """
    Process corpus with a spacy model.
    Args:
        documents: List of documents to be processed.
        nlp: Spacy model.

    Returns:
        sentences: List of tokenized sentences.
    """
    docs = list(nlp.pipe(documents, n_process=32))

    sentences = list()
    for doc in docs:
        for sent in doc.sents:
            new_sent = [t.lemma_ for t in sent if not t.is_stop]
            sentences.append(new_sent)
    return sentences


def process_corpus(documents, patterns, substrings):
    """
    Process a given collection of documents by cleaning and tokenizing each of them, combining them into a list
    of tokenized sentences.
    Args:
        documents: iterable or list(str) - Collection of documents to be processed.
        patterns: list of regex patterns to search in the documents.
        substrings: list of substrings to replace for each pattern in `patterns`.

    Returns: sentences - list(str) - A list containing the pre-processed and tokenized sentences from `documents`.
    """

    sentences = list()
    for _doc in documents:
        _doc = _doc.lower()
        for pattern, subs in zip(patterns, substrings):
            _doc = pattern.sub(subs, _doc.lower())

        # Tokenize document
        doc_sents = sent_tokenize(_doc.lower())
        for i, sent in enumerate(doc_sents):
            _sent = [t.lower() for t in word_tokenize(sent)]
            sentences.append(_sent)
    return sentences


def get_source_documents(con):
    query = "SELECT id, source, title, content, date(published_utc, 'unixepoch') as date, published_utc as timestamp " \
            " FROM newsdata " \
            " ORDER BY source"
    df = pd.read_sql(query, con)

    return df


def train_word2vec(df, source, patterns=[], substrings=[], nlp=None, **kwargs):

    if type(source) == str:
        d = df[df["source"] == source]
    elif type(source) == list:
        d = df[df["source"].isin(source)]

    if nlp is None:
        sentences = process_corpus(d["content"], patterns, substrings)
    else:
        d = d.sample(n=1000, replace=True, random_state=42)
        sentences = process_corpus_spacy(d["content"], nlp)
    model = None
    try:
        model = Word2Vec(sentences=sentences, **kwargs)
    except RuntimeError as e:
        print("Word2Vec error", e)

    return model


def exclude_articles(df: pd.DataFrame, exclude: set) -> pd.DataFrame:
    """
    Excludes from DataFrame df articles whose id are in `exclude`.
    Args:
        df: Corpus DataFrame.
        exclude: Set of article IDs to leave out.

    Returns:
        df_: Modified DataFrame without the articles in `exclude`.
    """
    df_ = df[~df["id"].isin(exclude)]
    print("- Exclude: %d before | %d now" % (len(df), len(df_)))
    return df_


def main():

    db_path = "../../data/nela/nela-covid.db"
    vocab_path = "../data/COVID_vocab.txt"
    output_path = "embeddings/source"

    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=None, help="Path to input NELA database")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to target vocab")
    parser.add_argument("--output", type=str, default=None, help="Path to output embeddings")
    parser.add_argument("--duplicates-file", dest="duplicates_file", type=str, default=None,
                        help="Path to the file of article duplicate pairs (csv).")
    parser.add_argument("--min-count", dest="min_count", type=int, default=100,
                        help="Minimum word count (for Word2Vec).")

    parser.add_argument("--source-groups", dest="source_groups", action="store_true",
                        help="Train embeddings for a specific group of sources")

    args = parser.parse_args()
    if args.db_path:
        db_path = args.db_path
    if args.vocab_path:
        vocab_path = args.vocab_path
    if args.output:
        output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if args.duplicates_file:
        duplicates = read_duplicates_file(args.duplicates_file)
        print("Found", len(duplicates), "duplicate articles.")
    else:
        duplicates = set()

    exclude = {*duplicates}  # Exclude these article ids from training

    vocab = read_vocabulary(vocab_path, split=False)
    patterns, substrings = get_patterns_and_substrings(vocab)  # get substitution patterns and strings for n-grams

    w2v_params = {"vector_size": 100,
                  "window": 10,
                  "min_count": args.min_count,
                  "workers": 64}

    con = sqlite3.connect(db_path)

    print("Reading data...")

    if not os.path.exists("corpus/corpus.df"):
        df = get_source_documents(con)
        if not os.path.exists("corpus"):
            os.mkdir("corpus")

        df.to_pickle("corpus/corpus.df")
    else:
        df = pd.read_pickle("corpus/corpus.df")

    df = exclude_articles(df, exclude)

    if not args.source_groups:
        sources = df["source"].unique()

        for s in sources:
            print("Training Word2Vec", s)
            model = train_word2vec(df, s, patterns, substrings, **w2v_params)
            if model:
                model.save(os.path.join(output_path, "%s.model" % s))
    else:
        output_path = "embeddings/clusters"
        try:
            os.makedirs(output_path)
        except Exception as e:
            print("File exists", e)
        sources_cluster_a = ["newswars", "davidicke", "newspunch", "infowars", "humansarefree",
                             "prisonplanet", "wakingtimes", "thedcclothesline"]
        sources_cluster_b = ["mercola", "sanevax", "junksciencecom", "healthyholisticliving", "althealthworks",
                             "allianceadvancedhealth"]
        sources_cluster_c = ["washingtonpost", "npr", "thehill", "vox", "usnews"]

        sources_ = {"cluster_a": sources_cluster_a, "cluster_b": sources_cluster_b, "cluster_c": sources_cluster_c}

        for c in sources_:
            print("Training word2Vec", sources_[c])
            nlp = spacy.load("en_core_web_sm")
            model = train_word2vec(df, sources_[c], patterns, substrings, nlp=nlp, **w2v_params)
            if model:
                model.save(os.path.join(output_path, "%s.model" % c))


if __name__ == "__main__":
    main()

