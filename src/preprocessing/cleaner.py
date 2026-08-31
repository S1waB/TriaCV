"""Resume text cleaning, normalization, tokenization, and lemmatization pipeline.

Technical terms like C++, .NET, Node.js, etc. are preserved through the
entire pipeline using a placeholder substitution strategy.
"""
import re
import string
import unicodedata
from typing import Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# NLTK resource bootstrapping
# ---------------------------------------------------------------------------

def ensure_nltk_resources() -> None:
    """Download required NLTK corpora and tokenizer data if not present."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet":   "wordnet",
        "tokenizers/punkt":     "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/omw-1.4":   "omw-1.4",
    }
    for data_path, package_name in resources.items():
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                pass


ensure_nltk_resources()


# ---------------------------------------------------------------------------
# Technical-term protection table
# Tokens are replaced by safe alphanumeric placeholders before any
# regex / tokenisation step, then restored afterwards.
# ---------------------------------------------------------------------------

TECH_TOKENS: Dict[str, str] = {
    # Languages with special punctuation
    r"(?<!\w)C\+\+(?!\w)":          "__CPP_LANG__",
    r"(?<!\w)C#(?!\w)":             "__CSHARP_LANG__",
    r"(?<!\w)F#(?!\w)":             "__FSHARP_LANG__",
    r"(?<!\w)\.NET(?!\w)":          "__DOTNET_TECH__",
    r"(?<!\w)ASP\.NET(?!\w)":       "__ASPNET_TECH__",
    r"(?<!\w)Node\.js(?!\w)":       "__NODEJS_TECH__",
    r"(?<!\w)Vue\.js(?!\w)":        "__VUEJS_TECH__",
    r"(?<!\w)React\.js(?!\w)":      "__REACTJS_TECH__",
    r"(?<!\w)Next\.js(?!\w)":       "__NEXTJS_TECH__",
    r"(?<!\w)Nuxt\.js(?!\w)":       "__NUXTJS_TECH__",
    r"(?<!\w)Express\.js(?!\w)":    "__EXPRESSJS_TECH__",
    r"(?<!\w)Three\.js(?!\w)":      "__THREEJS_TECH__",
    r"(?<!\w)D3\.js(?!\w)":         "__D3JS_TECH__",
    r"(?<!\w)TypeScript(?!\w)":     "__TYPESCRIPT_LANG__",
    r"(?<!\w)JavaScript(?!\w)":     "__JAVASCRIPT_LANG__",
    # Frameworks / ecosystems
    r"(?<!\w)scikit-learn(?!\w)":   "__SKLEARN_LIB__",
    r"(?<!\w)PyTorch(?!\w)":        "__PYTORCH_LIB__",
    r"(?<!\w)TensorFlow(?!\w)":     "__TENSORFLOW_LIB__",
    r"(?<!\w)Keras(?!\w)":          "__KERAS_LIB__",
    r"(?<!\w)OpenCV(?!\w)":         "__OPENCV_LIB__",
    r"(?<!\w)LangChain(?!\w)":      "__LANGCHAIN_LIB__",
    # Cloud / DevOps with special chars
    r"(?<!\w)CI/CD(?!\w)":          "__CICD_DEVOPS__",
    r"(?<!\w)ETL(?!\w)":            "__ETL_TECH__",
    r"(?<!\w)R(?!\w)":              "__R_LANG__",   # single-letter language
}

# Reverse map: placeholder -> display form
TECH_RESTORE: Dict[str, str] = {
    "__CPP_LANG__":       "c++",
    "__CSHARP_LANG__":    "c#",
    "__FSHARP_LANG__":    "f#",
    "__DOTNET_TECH__":    ".net",
    "__ASPNET_TECH__":    "asp.net",
    "__NODEJS_TECH__":    "node.js",
    "__VUEJS_TECH__":     "vue.js",
    "__REACTJS_TECH__":   "react.js",
    "__NEXTJS_TECH__":    "next.js",
    "__NUXTJS_TECH__":    "nuxt.js",
    "__EXPRESSJS_TECH__": "express.js",
    "__THREEJS_TECH__":   "three.js",
    "__D3JS_TECH__":      "d3.js",
    "__TYPESCRIPT_LANG__": "typescript",
    "__JAVASCRIPT_LANG__": "javascript",
    "__SKLEARN_LIB__":    "scikit-learn",
    "__PYTORCH_LIB__":    "pytorch",
    "__TENSORFLOW_LIB__": "tensorflow",
    "__KERAS_LIB__":      "keras",
    "__OPENCV_LIB__":     "opencv",
    "__LANGCHAIN_LIB__":  "langchain",
    "__CICD_DEVOPS__":    "ci/cd",
    "__ETL_TECH__":       "etl",
    "__R_LANG__":         "r",
}

# Keep protected tokens from stopword removal / lemmatisation
PROTECTED_RESTORED = set(TECH_RESTORE.values())


# ---------------------------------------------------------------------------
# Main cleaner class
# ---------------------------------------------------------------------------

class ResumeCleaner:
    """Robust cleaner for CV / Resume text with full NLP preprocessing pipeline.

    Pipeline steps
    --------------
    1. Unicode normalization (NFKD)
    2. Protect technical compound terms (C++, .NET, Node.js …)
    3. Strip URLs, emails, phone numbers
    4. Remove social artifacts (@mentions, #hashtags, RT)
    5. Remove remaining non-alphanumeric characters
    6. Restore protected tech terms
    7. Lowercase + split into tokens
    8. Remove stopwords (optional)
    9. Lemmatize (optional), skipping protected terms
    10. Rejoin tokens
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        language: str = "english",
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.language = language

        # Stopwords
        try:
            self.stop_words: set = set(stopwords.words(self.language))
        except Exception:
            # Minimal English fallback
            self.stop_words = {
                "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
                "she", "her", "it", "its", "they", "them", "their", "what", "which",
                "who", "this", "that", "these", "those", "am", "is", "are", "was",
                "were", "be", "been", "being", "have", "has", "had", "do", "does",
                "did", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                "of", "at", "by", "for", "with", "about", "to", "from", "in", "out",
                "on", "over", "under", "then", "when", "where", "how", "all", "any",
                "not", "only", "so", "than", "very", "can", "will", "just", "now",
            }
        # Remove single-char stopwords that clash with protected lang tokens
        self.stop_words.discard("r")
        self.stop_words.discard("c")

        # Lemmatizer
        try:
            self.lemmatizer: Optional[WordNetLemmatizer] = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _protect_tech_terms(text: str) -> str:
        """Replace technical compound tokens with safe placeholders."""
        for pattern, placeholder in TECH_TOKENS.items():
            text = re.sub(pattern, f" {placeholder} ", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _restore_tech_terms(text: str) -> str:
        """Restore placeholders back to canonical display forms."""
        for placeholder, display in TECH_RESTORE.items():
            text = text.replace(placeholder.lower(), display)
        return text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Apply the full cleaning pipeline to a raw resume string.

        Args:
            text: Raw resume text (any encoding artefacts should be resolved
                  upstream by the extractor).

        Returns:
            Space-joined string of clean, lemmatized tokens.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Unicode normalization
        text = unicodedata.normalize("NFKD", text)

        # 2. Protect technical compound terms
        text = self._protect_tech_terms(text)

        # 3. Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 4. Remove emails
        text = re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", " ", text)

        # 5. Remove phone numbers (international formats)
        text = re.sub(
            r"\b(?:\+?[\d]{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?)?\d{3,4}[\s.\-]?\d{4}\b",
            " ", text
        )

        # 6. Remove social artifacts
        text = re.sub(r"\b(?:RT|cc)\b", " ", text)
        text = re.sub(r"@\S+", " ", text)
        text = re.sub(r"#\S+", " ", text)

        # 7. Strip non-word characters (preserves underscores in placeholders)
        text = re.sub(r"[^\w\s]", " ", text)

        # 8. Lowercase
        text = text.lower()

        # 9. Restore tech placeholders to display forms
        text = self._restore_tech_terms(text)

        # 10. Tokenize (simple split to avoid NLTK punkt issues on custom tokens)
        tokens = text.split()

        # 11. Filter, remove stopwords, lemmatize
        cleaned_tokens: List[str] = []
        for token in tokens:
            # Strip residual edge punctuation
            if token not in PROTECTED_RESTORED:
                token = token.strip(string.punctuation + "_")
            if not token or len(token) < 2:
                continue
            if self.remove_stopwords and token in self.stop_words:
                continue
            if (
                self.lemmatize
                and self.lemmatizer is not None
                and token not in PROTECTED_RESTORED
            ):
                try:
                    token = self.lemmatizer.lemmatize(token)
                except Exception:
                    pass
            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens)

    def extract_keywords(self, text: str, top_k: int = 15) -> List[str]:
        """Return the top-k most frequent tokens from cleaned resume text.

        Args:
            text: Raw or pre-cleaned resume text.
            top_k: Number of top keywords to return.

        Returns:
            List of keyword strings ordered by frequency descending.
        """
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        if not tokens:
            return []
        freq: Dict[str, int] = {}
        for token in tokens:
            if len(token) >= 2:
                freq[token] = freq.get(token, 0) + 1
        sorted_kws = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kws[:top_k]]


if __name__ == "__main__":
    sample = (
        "Senior Software Engineer with 8 years experience in C++, .NET, Node.js and React.js. "
        "Email: john.doe@example.com | Phone: +1-555-123-4567 | https://johndoe.dev "
        "Expertise in scikit-learn, TensorFlow, CI/CD pipelines. Passionate about #OpenSource. "
        "@JohnDoe on GitHub."
    )
    cleaner = ResumeCleaner()
    result = cleaner.clean_text(sample)
    print("Input :", sample)
    print("Output:", result)
    print("Keywords:", cleaner.extract_keywords(sample))
