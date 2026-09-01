# decoding the char label shit
def build_vocab(alphabet):
    return sorted(alphabet.tolist())

# taking a text and giving indices for teh character labels
def encode_text(text, char_to_idx):
    indices = []
    for letter in text:
        if letter == ' ':
            letter = ''
        if letter not in char_to_idx:
            raise ValueError(f"Character {letter!r} not in vocabulary")
        indices.append(char_to_idx[letter])
    return indices

def decode_labels(labels, sorted_alphabet):
    return ''.join(sorted_alphabet[c] for c in labels)

# giving indices of all the characters in sorted alphabet
def build_char_to_idx(sorted_alphabet):
    return {char: idx for idx, char in enumerate(sorted_alphabet)}
