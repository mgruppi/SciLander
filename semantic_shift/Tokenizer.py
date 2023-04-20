import re


class WordTokenizer:
    def __init__(self, regex=None):
        if regex is None:
            self.regex = re.compile(r'\d+(?:[\,\.](?:\d+))'  # Numbers and digits (including ',' and '.').
                                    r'|(?:\w(?:\.\w)+(?:\.)?)'  # Acronyms (e.g.: U.S.)
                                    r'|\w+(?:[-_]\w+)*'  # Words and bi-grams separated by -
                                    r'|[\.\,\?\!\;]+'  # Matches punctuation
                                    r'|[\@\#\$\%]+')  # Matches special symbols

    def tokenize(self, text):
        return self.regex.findall(text)


class SentenceTokenizer:
    def __init__(self, regex=None):
        if regex is None:
            self.regex = re.compile(r'([\.\!\n\?])')