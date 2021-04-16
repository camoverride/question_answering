"""
Test the document retrieval system.
WARNING: Make sure that you have an internet connection before running this.
TODO: warm up model server.
"""

import unittest
from document_retrieval import get_articles


QUERY = "What is the capital of France?"


class TestDocumentRetrieval(unittest.TestCase):
    """
    Test the document retrieval function to make sure that plausible docs
    are being returned.
    """
    def test_get_articles(self):
        """
        Ping the search engine with various search terms to see what's returned
        """
        results = get_articles(QUERY, num_articles_search=2, characters_per_article=2500)

        # Check that 2 results were actually returned.
        assert len(results) == 2

        # Check that the titles are correct.
        assert results[0][0] == "Closed-ended question"
        assert results[1][0] == "Paris"

        # Check that the returned text is correct (search a subset)
        assert "On what day were you born?" in results[0][1]
        assert "Since the 17th century, Paris has been one of Europe's major" in results[1][1]
