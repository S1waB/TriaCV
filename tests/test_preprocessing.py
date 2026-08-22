"""Unit tests for ResumeCleaner and ResumeExtractor modules (Sprint 1)."""
import os
import pytest
from src.preprocessing.cleaner import ResumeCleaner
from src.preprocessing.extractor import ResumeExtractor


@pytest.fixture
def cleaner():
    return ResumeCleaner(remove_stopwords=True, lemmatize=True)


def test_cleaner_removes_urls_and_emails(cleaner):
    raw = "Contact me at dev@example.com or visit https://myportfolio.io for details."
    cleaned = cleaner.clean_text(raw)
    assert "dev@example.com" not in cleaned
    assert "https" not in cleaned
    assert "portfolio" in cleaned or "contact" in cleaned


def test_cleaner_preserves_technical_terms(cleaner):
    raw = "Expert in C++, C#, .NET Core, React.js and Node.js backend development."
    cleaned = cleaner.clean_text(raw)
    assert "c++" in cleaned
    assert "c#" in cleaned
    assert ".net" in cleaned


def test_cleaner_removes_stopwords(cleaner):
    raw = "This is a detailed summary of the main project with all the tasks."
    cleaned = cleaner.clean_text(raw)
    tokens = cleaned.split()
    assert "this" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "summary" in tokens
    assert "project" in tokens


def test_cleaner_empty_input(cleaner):
    assert cleaner.clean_text("") == ""
    assert cleaner.clean_text(None) == ""


def test_cleaner_keyword_extraction(cleaner):
    text = "Python Python Python machine learning scikit-learn PyTorch deep learning Python."
    kws = cleaner.extract_keywords(text, top_k=3)
    assert len(kws) > 0
    assert "python" in kws


def test_extractor_txt():
    content = "Senior Java Developer with 5 years experience."
    extracted = ResumeExtractor.extract_from_txt(content.encode("utf-8"))
    assert "Senior Java Developer" in extracted


def test_extractor_unsupported_fallback():
    extracted = ResumeExtractor.extract("Some raw text snippet")
    assert "Some raw text snippet" in extracted
