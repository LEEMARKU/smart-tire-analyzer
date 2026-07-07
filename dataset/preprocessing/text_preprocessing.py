"""
Text and Maintenance Record Preprocessing — Service reports, driver feedback, OCR text.
Implements ALL missing text preprocessing steps:
  1. Text cleaning (special chars, whitespace)
  2. Lowercasing
  3. Stop-word removal
  4. Tokenization
  5. Lemmatization
  6. Stemming
  7. Named Entity Recognition preprocessing
  8. Spell correction
"""

import re
import logging
from typing import List, Optional, Set, Callable

logger = logging.getLogger(__name__)

DEFAULT_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "when",
    "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "as", "until", "while", "about", "between",
    "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there",
}


def clean_text(text: str, remove_special: bool = True, collapse_whitespace: bool = True) -> str:
    """
    Clean text by removing special characters and normalizing whitespace.

    Args:
        text: Input text
        remove_special: Remove non-alphanumeric characters (except spaces and basic punctuation)
        collapse_whitespace: Collapse multiple spaces into one

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    cleaned = text.strip()

    if remove_special:
        cleaned = re.sub(r"[^a-zA-Z0-9\s\.\,\;\:\!\?\-]", " ", cleaned)

    if collapse_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_stopwords(
    tokens: List[str],
    stop_words: Optional[Set[str]] = None,
    additional_stop_words: Optional[List[str]] = None,
) -> List[str]:
    """
    Remove stop words from token list.

    Args:
        tokens: List of tokens
        stop_words: Set of stop words (uses DEFAULT_STOP_WORDS if None)
        additional_stop_words: Additional domain-specific stop words

    Returns:
        Filtered token list
    """
    if stop_words is None:
        stop_words = DEFAULT_STOP_WORDS.copy()

    if additional_stop_words:
        stop_words = stop_words | set(w.lower() for w in additional_stop_words)

    return [t for t in tokens if t.lower() not in stop_words]


def tokenize_text(text: str, method: str = "whitespace", remove_punct: bool = True) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Input text
        method: 'whitespace' (split on whitespace), 'regex' (word boundaries),
                'nltk' (NLTK tokenizer if available)
        remove_punct: Remove punctuation tokens

    Returns:
        List of tokens
    """
    if not text:
        return []

    if method == "whitespace":
        tokens = text.split()
    elif method == "regex":
        tokens = re.findall(r"\b\w+\b", text)
    elif method == "nltk":
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
        except ImportError:
            logger.warning("NLTK not available, falling back to regex tokenization")
            tokens = re.findall(r"\b\w+\b", text)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")

    if remove_punct:
        tokens = [t for t in tokens if re.search(r"[a-zA-Z0-9]", t)]

    return tokens


def lemmatize_text(tokens: List[str], pos: str = "n") -> List[str]:
    """
    Lemmatize tokens to base form.

    Args:
        tokens: List of tokens
        pos: Part of speech ('n' noun, 'v' verb, 'a' adjective, 'r' adverb)

    Returns:
        Lemmatized tokens
    """
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token, pos=pos) for token in tokens]
    except ImportError:
        logger.warning("NLTK WordNetLemmatizer not available, returning original tokens")
        return tokens


def stem_tokens(tokens: List[str], method: str = "porter") -> List[str]:
    """
    Stem tokens to root form.

    Args:
        tokens: List of tokens
        method: 'porter' (Porter stemmer), 'snowball' (Snowball stemmer),
                'lancaster' (Lancaster stemmer)

    Returns:
        Stemmed tokens
    """
    try:
        from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
        stemmers = {
            "porter": PorterStemmer(),
            "snowball": SnowballStemmer("english"),
            "lancaster": LancasterStemmer(),
        }
        stemmer = stemmers.get(method)
        if stemmer is None:
            raise ValueError(f"Unknown stemming method: {method}")
        return [stemmer.stem(token) for token in tokens]
    except ImportError:
        logger.warning("NLTK stemmer not available, returning original tokens")
        return tokens


def preprocess_for_ner(
    text: str,
    preserve_case: bool = False,
) -> str:
    """
    Preprocess text for Named Entity Recognition.
    Preserves capitalization for proper noun detection.

    Args:
        text: Input text
        preserve_case: If True, keep original case for NER

    Returns:
        NER-ready text
    """
    if not isinstance(text, str):
        return ""

    cleaned = re.sub(r"[^\w\s\.\#\@]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not preserve_case:
        cleaned = cleaned.lower()

    return cleaned


def spell_correct_text(
    text: str,
    dictionary: Optional[Set[str]] = None,
    max_distance: int = 2,
) -> str:
    """
    Basic spell correction using Levenshtein distance.

    Args:
        text: Input text
        dictionary: Set of known words (auto from common tire terms if None)
        max_distance: Maximum edit distance for correction

    Returns:
        Spell-corrected text
    """
    TIRE_TERMS = {
        "tire", "tyre", "tread", "groove", "sidewall", "pressure", "psi",
        "bar", "kpa", "temperature", "vibration", "alignment", "balance",
        "rotation", "inflation", "deflation", "puncture", "blowout",
        "radial", "bias", "ply", "belt", "valve", "rim", "wheel",
        "hub", "axle", "suspension", "steering", "brake", "traction",
        "michelin", "bridgestone", "goodyear", "continental", "pirelli",
        "dunlop", "yokohama", "hankook", "sumitomo", "toyo",
    }

    if dictionary is None:
        dictionary = TIRE_TERMS

    words = text.split()
    corrected = []

    for word in words:
        if word.lower() in dictionary:
            corrected.append(word)
            continue
        best_match = word
        best_dist = max_distance + 1
        for dict_word in dictionary:
            dist = _levenshtein_distance(word.lower(), dict_word.lower())
            if 0 < dist < best_dist:
                best_dist = dist
                best_match = dict_word
        corrected.append(best_match)

    return " ".join(corrected)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def preprocess_text_pipeline(
    text: str,
    do_clean: bool = True,
    do_lowercase: bool = True,
    do_tokenize: bool = True,
    do_remove_stopwords: bool = True,
    do_lemmatize: bool = False,
    do_stem: bool = False,
    do_ner_prep: bool = False,
    do_spell_correct: bool = False,
) -> dict:
    """
    Complete text preprocessing pipeline.

    Args:
        text: Input text
        do_clean: Apply text cleaning
        do_lowercase: Apply lowercasing
        do_tokenize: Apply tokenization
        do_remove_stopwords: Apply stop-word removal
        do_lemmatize: Apply lemmatization
        do_stem: Apply stemming
        do_ner_prep: Apply NER preprocessing
        do_spell_correct: Apply spell correction

    Returns:
        dict with keys: 'cleaned', 'tokens', 'lemmatized', 'stemmed',
                       'ner_ready', 'spell_corrected'
    """
    result = {"original": text}

    if do_ner_prep:
        result["ner_ready"] = preprocess_for_ner(text)
        return result

    cleaned = text
    if do_clean:
        cleaned = clean_text(cleaned)
    if do_lowercase:
        cleaned = lowercase_text(cleaned)
    result["cleaned"] = cleaned

    if do_spell_correct:
        result["spell_corrected"] = spell_correct_text(cleaned)
        cleaned = result["spell_corrected"]

    tokens = tokenize_text(cleaned) if do_tokenize else cleaned.split()
    result["tokens"] = tokens

    if do_remove_stopwords:
        result["no_stopwords"] = remove_stopwords(tokens)

    if do_lemmatize:
        result["lemmatized"] = lemmatize_text(tokens)

    if do_stem:
        result["stemmed"] = stem_tokens(tokens)

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    sample = "The tire tread depth is 3.2mm and the sidewall shows sign of cracking."
    result = preprocess_text_pipeline(sample)
    logger.info(f"Original: {sample}")
    logger.info(f"Cleaned: {result['cleaned']}")
    logger.info(f"Tokens: {result['tokens']}")
    logger.info(f"No stopwords: {result.get('no_stopwords', [])}")
