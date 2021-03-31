from typing import List
import wikipedia as wiki


def get_articles(query: str, num_articles: int, characters_per_article: int) -> List[tuple]:
    """
    This function takes a query and downloads the text if relevant Wikipedia articles.

    Parameters
    ----------
    query : str
        A query that will be used to identify relevant wikipedia articles.
        Example: "Who did Joe Biden defeat in 2020?"
    num_articles : int
        The number of articles that will be searched and downloaded.
    characters_per_article : int
        The number of characters that will be included. The answer to a question
        is usually in the beginning of an article, so it's not necessary to search
        the entire article.

    Returns
    -------
    list
        A list of tuples where an article's title is mapped to its text. Example: 
        [("United States", "The United States is..."), ("Barack Obama", "Barack Obama is a politician..."), ...]
    """

    # A list of article titles - these may not be the "correct" titles (see below)
    article_titles = wiki.search(query, results=num_articles)

    # Collect tuples of (article_title, article_text)
    article_data = []
    for title in article_titles:
        try:
            # Try to get the text of the article.
            text = wiki.page(title).content[:characters_per_article]
        except wiki.exceptions.PageError as e:
            # Not all the results returned by wiki.search are valid titles.
            # wiki.suggest returns a valid title for the "incorrect" title
            # i.e. "Joe Biden" -> "joe biden n"
            title = wiki.suggest(title)
            text = wiki.page(title).content[:characters_per_article]
        
        article_data.append((title, text))

    return article_data
