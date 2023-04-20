import numpy as np
from collections import defaultdict
from scipy.linalg import orthogonal_procrustes


class WordVectorGroup:
    def __init__(self, vocabs, embeddings, labels=None, min_count=10):
        """
        Creates a series of word vectors from a group of vocab and embeddings sources.
        Vocabs are unified into a single vocabulary V_t. If a token `t` does not exist in a given source,
        its corresponding vector will be a zero vector.
        This can be used to jointly hold embeddings for a group of sources(domains) or a time series of embeddings.
        Args:
            vocabs: list(str) Vocabulary of each source.
            embeddings: list(np.ndarray) Embedding matrices as a numpy array of dimensions |V| x d where V is the
            vocabulary and d is the embedding dimension.
            labels: list(str) [optional] Labels for each source.
            min_count: (int) Determines the minimum number of occurrences of a word across sources in order for it
            not to be discarded
        """

        # Create unified vocabulary
        word_count = defaultdict(int)
        for voc in vocabs:
            for word in voc:
                word_count[word] += 1

        # Construct vocabulary with mappings for id->word and word->id
        vocab = {w for w in word_count if word_count[w] >= min_count}
        self.idx_to_word = np.array(sorted(vocab))
        self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}
        self.emb = np.zeros((len(embeddings), len(self.idx_to_word), embeddings[0].shape[1]), dtype=np.float)

        if labels:
            self.labels = np.array(labels)
        else:
            self.labels = np.arange(0, len(vocabs))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

        for i, x in enumerate(embeddings):
            for j, word in enumerate(vocabs[i]):
                if word in vocab:  # Only add word if it is in the valid vocab (after pruning min_counts)
                    self.emb[i][self.word_to_idx[word]] = embeddings[i][j]

    def __len__(self):
        return len(self.emb)

    def __contains__(self, word):
        return word in self.word_to_idx

    def get_valid_vocab(self, group_id):
        """
        Get valid vocabulary for the given `group_id`.
        The valid vocabulary consists of words whose embedding entries are not zero row vectors.
        Args:
            group_id: (int) Index of the group.

        Returns:
            vocab (np.ndarray) - array of valid vocabulary for `group_id`.
        """
        if group_id >= len(self.emb) or group_id < 0:
            print("Error: invalid group index")
            return None
        non_zero = ~np.all(self.emb[group_id] == 0, axis=1)
        return self.idx_to_word[non_zero]

    def get_valid_indices(self, group_id):
        """
        Similar to `get_valid_vocab` but return word indices instead of words.
        Args:
            group_id: (int) Index of the roup.

        Returns:
            indices (np.ndarray) - array of indices.
        """
        if group_id >= len(self.emb) or group_id < 0:
            print("Error: invalid group index")
            return None
        indices = np.where(~np.all(self.emb[group_id] == 0, axis=1))[0]
        return indices

    def get_non_zero_mask(self, idx):
        """
        Returns the binary mask vector for words whose vectors are non-zero.
        Returns:
            mask (np.array) Array of bool where mask[i] == True iff self.emb[idx] is non-zero.
        """
        return ~np.all(self.emb[idx] == 0, axis=1)

    def get_group_label(self, idx):
        """
        Retrieves the label assigned to group of index `idx`.
        Args:
            idx: (int) Index of the group.

        Returns:
            label (str), if label exists. Otherwise, returns `None`.
        """
        if self.labels:
            return self.labels[idx]
        else:
            return None

    def get_label_mask(self, labels):
        """
        Selects a set of groups matching the labels in `labels` and returns the corresponding binary mask.
        Args:
            labels: (array-like) - The labels to search for.
        Returns:
            mask (np.ndarray) - Binary mask of matching elements
        """
        return np.isin(self.labels, labels)

    def get_word_mask(self, words):
        """
        Creates a binary mask for the words in `words`, if they exists.
        The result is a binary mask m (x0, x1, ..., xn) where xi = True if self.id_to_word[i] in `words`.
        Args:
            words: (array-like) Words to get indices for.

        Returns:
            m (np.ndarray) - Binary mask of matching words.
        """
        return np.isin(self.idx_to_word, words)

    def align_all(self, anchor, indices):
        """
        Align all embeddings in the group to `anchor` using rows from `indices`.
        `anchor` must be a n x d matrix and `indices` must be a 1-d array of length `n` containing the row
        indices to be used in alignment.
        The indices correspond to the words to be used as anchors.
        Zero vectors are not used in alignment and are discarded if found in an embedding.
        Args:
            anchor - 2-d matrix of anchor points
            indices - the list of indices of embeddings to be aligned to the anchors.
        """

        if len(anchor) != len(indices):
            print("*Anchor dim", anchor.shape, "does not match indices", len(indices))
            return

        print(anchor.shape)
        print(len(indices))

        for i, x in enumerate(self.emb):
            x_filtered = x[indices]  # Filter relevant indices first
            # Filter all-zero entries. We do not want to align on those.
            idx_mask = ~np.all(x_filtered == 0, axis=1)
            print("Aligning", self.labels[i], sum(idx_mask))
            q, _ = orthogonal_procrustes(x_filtered[idx_mask], anchor[idx_mask])
            self.emb[i] = np.dot(x, q)

    def group_size(self):
        return len(self.emb)

    def vocab_size(self):
        return len(self.idx_to_word)


class WordFrequencyGroup:
    def __init__(self, word_counts, labels=None, min_source_count=10):
        """
        Create a WordFrequencyGroup object using dictionaries of word_counts, labels and min count arguments.
        Args:
            word_counts: List of dictionaries containing word counts for wc in word_counts: wc maps str -> int
            labels: Labels for the sources. If None, sources are labeled by their index (0 to n).
            min_source_count: Minimum number of sources a word needs to appear in in order to be kept.
        """
        # Create unified vocabulary
        word_source_count = defaultdict(int)
        for voc in word_counts:
            for word in word_counts[voc]:
                word_source_count[word] += 1

        # Construct vocabulary with mappings for id->word and word->id
        vocab = {w for w in word_source_count if word_source_count[w] >= min_source_count}
        self.idx_to_word = np.array(sorted(vocab))
        self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}
        self.counts = np.zeros((len(word_counts), len(self.idx_to_word)), dtype=float)

        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = np.arange(0, len(word_counts))

        for i, wcount in enumerate(word_counts):
            print(wcount)
            self.counts[i] = [word_counts[wcount][w] if w in word_counts[wcount] else 0 for w in self.idx_to_word]


# Create aliases for WordVectorGroup
WordVectorSeries = WordVectorGroup
TimeWordVectors = WordVectorSeries
