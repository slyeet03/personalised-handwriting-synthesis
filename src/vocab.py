def build_vocab(alphabet):
    return sorted(alphabet.tolist())

def encode_text(text, char_to_idx):
    indices = []
    for letter in text:
        if letter not in char_to_idx:
            raise ValueError(f"Character {letter!r} not in vocabulary")
        indices.append(char_to_idx[letter])
    return indices

def decode_labels(labels, sorted_alphabet):
    return ''.join(sorted_alphabet[c] for c in labels)

def build_char_to_idx(sorted_alphabet):
    return {char: idx for idx, char in enumerate(sorted_alphabet)}
