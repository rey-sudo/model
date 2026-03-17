from collections import Counter

def reconstruir_frase(clasificacion):
    prefix, frases_scores = clasificacion
    # add temperature
    # obtener frases válidas
    sentences = [s for s in frases_scores.keys() if s.strip()]

    split_sentences = [s.split() for s in sentences]
    max_len = max(len(s) for s in split_sentences)

    result = []

    for i in range(max_len):
        words = [s[i] for s in split_sentences if len(s) > i]
        word = Counter(words).most_common(1)[0][0]
        result.append(word)

    return result, " ".join(result)