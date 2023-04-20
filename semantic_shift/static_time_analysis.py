import pickle
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
from mapping import read_vocabulary
from WordVectorGroup import WordVectorSeries
from WordVectors import WordVectors
import matplotlib.pyplot as plt
import os


def get_row_aligned_matrices(w2id_a, vecs_a, w2id_b, vecs_b):
    """
    Returns the row-aligned matrices of vecs_a and vecs_b.
    Row-aligned matrices are embedding matrices such that the i-th row in vecs_a and vecs_b are associated with the same
    word.
    This is done by finding the intersection between `w2id_a` and `w2id_b` and constructing an aligned matrix `a`
    using those words.
    Args:
        w2id_a: Word-to-id dict from embedding A.
        vecs_a: Embedding matrix A.
        w2id_b: Word-to-id dict from embedding B.
        vecs_b: Embedding matrix B.

    Returns:
        a - list(str) words, numpy.ndarray a, numpy.ndarray b - `words` is a list of the common words, `a` and `b`
        are the row-aligned matrices.
    """
    common_vocab = set.intersection(set(w2id_a.keys()), set(w2id_b.keys()))
    words = np.array(sorted(common_vocab))
    a = np.zeros((len(words), vecs_a.shape[1]), dtype=np.float32)
    b = np.zeros((len(words), vecs_b.shape[1]), dtype=np.float32)

    for i, w in enumerate(words):
        a[i] = vecs_a[w2id_a[w]]
        b[i] = vecs_b[w2id_b[w]]

    return words, a, b


def get_row_aligned_indices(w2id_a, w2id_b):
    """
    Returns a pair of lists of indices i_a, i_b such that the index of the j-th position in both correspond to the
    same word in each embedding.
    Args:
        w2id_a: Word to index list of embedding a.
        w2id_b: Word to index list of embedding b.

    Returns:
        i_a, i_b: np.array of row aligned indices.
    """
    common_vocab = set.intersection(set(w2id_a.keys()), set(w2id_b.keys()))
    words = sorted(common_vocab)

    i_a = np.array([w2id_a[w] for w in words], dtype=int)
    i_b = np.array([w2id_b[w] for w in words], dtype=int)

    return i_a, i_b


def align_to_anchor(embedding_series, anchor):
    """
    Align all embeddings in WordVectorSeries `embedding_series` to WordVectors `anchor`.
    Args:
        embedding_series (WordVectorSeries): `WordVectorSeries` as an embedding series.
        anchor (WordVectors): WordVectors to use as anchor.
    Returns:
    """

    # Get indices from A and B matching the same words. I.e.: i_a[i] corresponds to the same word as i_b[i]
    i_a, i_b = get_row_aligned_indices(embedding_series.word_to_idx, anchor.word_id)

    # print("A", len(embedding_series.id_to_word))
    # print("B", len(anchor.words))
    # print("xsec", len(i_a))

    for i, e in enumerate(embedding_series.emb):
        m_a = e[i_a]
        # Rows of zeros indicate absence of that word in the source
        # We will make sure that zero-row-vectors are not included in the alignment
        a_zeros = np.all(m_a == 0, axis=1)
        a_indices = i_a[~a_zeros]
        b_indices = i_b[~a_zeros]

        Q, _ = orthogonal_procrustes(e[a_indices], anchor.vectors[b_indices])
        embedding_series.emb[i] = np.dot(e, Q)


def load_cord19(path=None):
    """
    Loads Cord19 embeddings.
    Returns:
        wv (WordVectors): Cord-19 embeddings.

    """
    if path is None:
        path = "embeddings/static/cord19.model"
    model = Word2Vec.load(path)
    wv = WordVectors(words=model.wv.index_to_key, vectors=model.wv.vectors)

    return wv


def get_word_time_series(word, data, anchor):
    x = np.array([cosine(e[data.word_to_idx[word]], anchor[word]) for e in data.emb], dtype=np.float32)
    return x


def main():
    path_rel = "embeddings/static/time_reliable.pickle"
    path_unr = "embeddings/static/time_unreliable.pickle"

    data_cord19 = load_cord19()

    with open(path_rel, "rb") as fin:
        data_rel = pickle.load(fin)
    with open(path_unr, "rb") as fin:
        data_unr = pickle.load(fin)

    align_to_anchor(data_rel, data_cord19)
    align_to_anchor(data_unr, data_cord19)

    vocab_path = "../data/COVID_vocab.txt"
    # Read target vocabulary
    tgt_vocab = read_vocabulary(vocab_path, split=False, sep="_")  # read vocabulary as a set().

    common_vocab = sorted([w for w in tgt_vocab if w in data_rel and w in data_unr and w in data_cord19])

    print(common_vocab)
    t = np.arange(len(data_rel.emb))

    path_result = "results/static"
    if not os.path.exists(path_result):
        os.makedirs(path_result)

    for w in common_vocab:
        x_r = get_word_time_series(w, data_rel, data_cord19)
        x_u = get_word_time_series(w, data_unr, data_cord19)
        fig, ax = plt.subplots()
        ax.plot(t, x_r, label="reliable")
        ax.plot(t, x_u, label="unreliable")

        ax.set_title(w)
        ax.set_xlabel("Week")
        ax.set_ylabel("Semantic distance")
        ax.legend()
        fig.savefig("%s.png" % os.path.join(path_result, w))
        fig.clf()


if __name__ == "__main__":
    main()
