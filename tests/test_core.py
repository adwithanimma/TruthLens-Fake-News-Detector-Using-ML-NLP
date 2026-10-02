"""Unit tests for preprocessing and article extraction. Run: python -m pytest tests -q"""

from __future__ import annotations

import pytest

from src.extractor import ExtractionError, extract_article
from src.text_utils import clean_text, preprocess, remove_stopwords


class TestCleanText:
    def test_lowercases_and_strips_punctuation(self):
        assert clean_text("Hello, WORLD!!!") == "hello world"

    def test_removes_urls(self):
        out = clean_text("Breaking https://example.com/x?a=1 news")
        assert "example.com" not in out
        assert "breaking" in out and "news" in out

    def test_collapses_whitespace(self):
        assert clean_text("a\n\n  b\t c") == "a b c"

    def test_keeps_contractions(self):
        assert "don't" in clean_text("Don't panic")

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""


class TestStopwords:
    def test_removes_common_words(self):
        out = remove_stopwords("the quick brown fox is over there")
        assert "the" not in out.split()
        assert "quick" in out and "brown" in out

    def test_preprocess_is_idempotent(self):
        text = "The Senate voted on the bill and the bill passed."
        assert preprocess(preprocess(text)) == preprocess(text)

    def test_preprocess_removes_all_stopwords(self):
        assert preprocess("the of and to in") == ""


class TestExtractionValidation:
    def test_empty_url(self):
        with pytest.raises(ExtractionError, match="No URL"):
            extract_article("")

    @pytest.mark.parametrize("url", ["not-a-url", "example.com", "ftp://example.com/f", "javascript:alert(1)"])
    def test_invalid_scheme_or_host(self, url):
        with pytest.raises(ExtractionError, match="Invalid URL"):
            extract_article(url)

    def test_network_failure_is_wrapped(self, monkeypatch):
        import requests

        def boom(*args, **kwargs):
            raise requests.exceptions.ConnectionError("no route")

        monkeypatch.setattr("src.extractor.requests.get", boom)
        with pytest.raises(ExtractionError, match="Could not fetch"):
            extract_article("https://example.com/a")

    def test_timeout_is_wrapped(self, monkeypatch):
        import requests

        def boom(*args, **kwargs):
            raise requests.exceptions.Timeout("slow")

        monkeypatch.setattr("src.extractor.requests.get", boom)
        with pytest.raises(ExtractionError, match="too long"):
            extract_article("https://example.com/a")


class TestExtractionParsing:
    @pytest.fixture
    def html(self):
        return """
        <html><head>
          <title>Sample News</title>
          <meta property="og:title" content="Council Approves Budget">
          <meta property="og:image" content="https://example.com/img.jpg">
          <meta property="article:published_time" content="2026-03-01T09:00:00Z">
        </head><body>
          <nav>Home About Contact</nav>
          <article>
            <p>The council voted on Tuesday to approve the annual budget after several
               weeks of public consultation and review by the finance committee.</p>
            <p>Members debated the allocation for transport infrastructure, with the
               final figure settling slightly above the draft proposal.</p>
            <p>Residents attending the meeting asked about the timetable for road
               repairs, and staff said a schedule would be published next month.</p>
            <p>The mayor said the decision reflected priorities set out during the
               election campaign earlier in the year for essential services.</p>
          </article>
          <script>console.log('tracking pixel')</script>
        </body></html>
        """

    @pytest.fixture
    def mock_page(self, monkeypatch, html):
        class Resp:
            url = "https://news.example.com/story"
            text = html

            def raise_for_status(self):
                return None

        monkeypatch.setattr("src.extractor.requests.get", lambda *a, **k: Resp())
        return Resp

    def test_extracts_title_and_metadata(self, mock_page):
        art = extract_article("https://news.example.com/story")
        assert art.title == "Council Approves Budget"
        assert art.source == "news.example.com"
        assert art.image.endswith("img.jpg")
        assert art.published.startswith("2026-03-01")

    def test_body_excludes_nav_and_script(self, mock_page):
        art = extract_article("https://news.example.com/story")
        assert "council voted on Tuesday" in art.text
        assert "Home About Contact" not in art.text
        assert "tracking pixel" not in art.text

    def test_word_count_matches_text(self, mock_page):
        art = extract_article("https://news.example.com/story")
        assert art.word_count == len(art.text.split())
        assert art.paragraphs

    def test_thin_page_raises(self, monkeypatch):
        class Resp:
            url = "https://example.com/x"
            text = "<html><body><div><p>Too short.</p></div></body></html>"

            def raise_for_status(self):
                return None

        monkeypatch.setattr("src.extractor.requests.get", lambda *a, **k: Resp())
        with pytest.raises(ExtractionError, match="Could not extract"):
            extract_article("https://example.com/x")