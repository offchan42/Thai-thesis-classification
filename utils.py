def pretty_trim(text):
    words = text.split(u'|')
    stripped_words_generator = (word.strip() for word in words)
#     stemmed_words_generator = (stemmer.stem(word) for word in stripped_words_generator)
    trimmed_words = (word for word in stripped_words_generator if 1 < len(word)) # retains words that are not empty
    alpha_words = (word for word in trimmed_words if not word.isnumeric() or len(word) <= 4) # allow only <= 4-digit number
    return u' '.join(alpha_words)

def simple_split(string):
    return string.split()
