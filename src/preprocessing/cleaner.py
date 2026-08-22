"""Resume text cleaning, normalization, and tokenization/lemmatization pipeline."""
import re
import string
import unicodedata
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
def ensure_nltk_resources():
    """Download required NLTK resources safely."""
    for resource in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource in ["stopwords", "wordnet"] else f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


ensure_nltk_resources()


class ResumeCleaner:
    """Robust cleaner for CV/Resume text data with lemmatization and stopword removal."""

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True, language: str = "english"):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.language = language

        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words(self.language))
        except Exception:
            self.stop_words = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
                "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
                "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
                "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
                "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
                "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
                "about", "against", "between", "into", "through", "during", "before", "after",
                "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                "under", "again", "further", "then", "once", "here", "there", "when", "where",
                "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
                "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
            }

        # Initialize Lemmatizer
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """Perform full text cleaning pipeline on raw resume string.

        Steps:
        1. Unicode normalization (NFKD)
        2. Remove URLs, emails, phone numbers
        3. Remove mentions (@) and hashtags (#)
        4. Remove non-ASCII / special bullet characters
        5. Convert to lowercase
        6. Remove punctuation (while protecting skills like c++, .net)
        7. Remove stopwords & apply lemmatization
        8. Strip extra whitespace
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Unicode normalization
        text = unicodedata.normalize("NFKD", text)

        # 2. Preserve specific technical keywords before punct stripping
        text = re.sub(r"(?i)(?<!\w)C\+\+(?!\w)", " cpp_lang ", text)
        text = re.sub(r"(?i)(?<!\w)C#(?!\w)", " csharp_lang ", text)
        text = re.sub(r"(?i)(?<!\w)\.NET(?!\w)", " dotnet_tech ", text)
        text = re.sub(r"(?i)(?<!\w)Node\.js(?!\w)", " nodejs_tech ", text)
        text = re.sub(r"(?i)(?<!\w)Vue\.js(?!\w)", " vuejs_tech ", text)
        text = re.sub(r"(?i)(?<!\w)React\.js(?!\w)", " reactjs_tech ", text)

        # 3. Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 4. Remove emails
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

        # 5. Remove phone numbers & date patterns
        text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{3,5}\b", " ", text)

        # 6. Remove mentions, hashtags, RT
        text = re.sub(r"\b(?:RT|cc)\b", " ", text)
        text = re.sub(r"@\S+", " ", text)
        text = re.sub(r"#\S+", " ", text)

        # 7. Remove non-alphanumeric special characters and symbols
        text = re.sub(r"[^\w\s]", " ", text)

        # Restore tech tokens
        text = text.replace("cpp_lang", "c++")
        text = text.replace("csharp_lang", "c#")
        text = text.replace("dotnet_tech", ".net")
        text = text.replace("nodejs_tech", "node.js")
        text = text.replace("vuejs_tech", "vue.js")
        text = text.replace("reactjs_tech", "react.js")

        # 8. Lowercase & tokenization
        text = text.lower()
        words = text.split()

        cleaned_tokens: List[str] = []
        protected_tech = {"c++", "c#", ".net", "react.js", "node.js", "vue.js", "asp.net", "r", "c"}
        for word in words:
            if word in protected_tech:
                w = word
            else:
                # Strip remaining punctuation from edges
                w = word.strip(string.punctuation)
            if not w or (len(w) <= 1 and w not in ["c", "r"]):
                continue
            if self.remove_stopwords and w in self.stop_words:
                continue
            if self.lemmatize and self.lemmatizer and w not in protected_tech:
                try:
                    w = self.lemmatizer.lemmatize(w)
                except Exception:
                    pass
            cleaned_tokens.append(w)

        return " ".join(cleaned_tokens)

    def extract_keywords(self, text: str, top_k: int = 15) -> List[str]:
        """Extract top recurring keywords from resume text."""
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        if not tokens:
            return []
        freq = {}
        for token in tokens:
            if len(token) > 2:
                freq[token] = freq.get(token, 0) + 1
        sorted_kws = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_kws[:top_k]]
