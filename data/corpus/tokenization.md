# Subword Tokenization

Language models operate on token ids, not raw characters. Subword tokenizers such as
byte-pair encoding (BPE) and WordPiece build a vocabulary by iteratively merging the
most frequent symbol pairs in a training corpus, so common words become single tokens
while rare words decompose into meaningful pieces.

This balances two failure modes: a word-level vocabulary cannot represent unseen
words, and a character-level vocabulary makes sequences impractically long. Byte-level
BPE, used by the GPT family, guarantees any string can be encoded by falling back to
raw bytes.

Tokenization quirks have practical consequences. Numbers often split inconsistently,
which hurts arithmetic; whitespace handling differs across tokenizers; and languages
underrepresented in the training corpus fragment into many more tokens, raising cost
and degrading quality. Vocabulary sizes typically range from 32k to 256k entries.
