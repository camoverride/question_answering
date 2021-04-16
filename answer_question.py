import logging
logging.basicConfig(level=logging.INFO)
logging.info("Running QA module")

from typing import List

from document_retrieval import get_articles
from reading_comprehension import get_model_predictions


class BertTokenSizeOutOfRange(Exception):
    """
    Raised when you are attempting to send too much data to the BERT Model.
    BERT can embed at most 512 tokens. Because BERT QA uses 3 special tokens,
    the max length for a query + context is 509 tokens. This error should be
    raised whenever len(query + context) > 509.

    Ideally, this error should be raised only after the query + context are
    tokenized by BERT's tokenizer. However, this operation is expensive so
    it may be advisable to measure simple whitespace tokenized items instead.

    Attributes:
        token_count -- the length of a list of word tokens which is too great.
    """
    def __init__(self, token_count):
        self.token_count = token_count


class Answerer:
    """
    Attributes
    ----------
    model_server_address : str
        Address of the BERT model server.
    num_articles_search : int
        The number of articles that will be downloaded by the document retriever.
    characters_per_article : int
        Only use the first `characters_per_article` from the article - the answer is likely to
        be in the beginning of a document.

    Methods
    -------
    answer_question
        Returns an answer object in response to a question.
        This answer must be parsed like ans["answer"]["answer"]
    """

    def __init__(self, model_server_address, num_articles_search=5, characters_per_article=2500):
        self.model_server_address = model_server_address
        self.num_articles_search = num_articles_search
        self.characters_per_article = characters_per_article


    def _get_tokens(self, query_or_context: str) -> int:
        """
        TODO: currently this just splits on whitespace. However, it should ideally use
        BERT's tokenizer. However, this is computationally expensive to do.

        Parameters
        ----------
        query_or_context : str
            A string of text that will be tokenized.    

        Returns
        -------
        int
            The number of tokens in the string.
        """
        return query_or_context.split(" ")


    def _get_article_chunks(self, article: str) -> List[str]:
        """
        This function divides an article into smaller chunks. BERT has an upper limit for
        the number of tokens it can accept, so the chunk size must be accordingly small.
        BERT can accept 509 tokens (512 minus the three special tokens). So the length of
        the question + answer must be <= 509.
        TODO: chunks should have some overlap in case the answer exists at a boundary.

        Parameters
        ----------
        article : str
            An article that will be chunked
        
        Returns
        -------
        list
            A list of strings, which are chunks of the article.
        """
        tokens = self._get_tokens(article)
        for i in range(0, len(tokens), self.chunk_size):
            chunk = " ".join(tokens[i:i + self.chunk_size])

            yield chunk


    def _decider(self, model_evaluations: List[dict], question: str) -> dict:
        """
        This function accepts a list of dicts, where each dict contains info about the model's
        output for a different chunk. This function compares them and decides which answer is
        best. The criterion is simple: the best answer is the answer with the highest value
        for start_scores_max + end_scores_max
        TODO: find a better decision criteria, looking at the entire dist possibly

        Parameters
        ----------
        model_evaluations : list
            A list of dicts, where the dict contains information about the model's
            evaluation. See `answer question` where the schema of this dict is defined.

        Returns
        -------
        dict
            A dict where the model evaluation containing the answer is mapped to the key
            "answer" and the other evaluations are in a list and mapped to "other results"
        """
        choice_index, choice_max_sum = 0, 0

        for i, evaluation in enumerate(model_evaluations):
            sum_scores = evaluation["start_scores_max"] + evaluation["end_scores_max"]
            if sum_scores > choice_max_sum:
                choice_max_sum = sum_scores
                choice_index = i

        answer = model_evaluations[choice_index]
        model_evaluations.pop(choice_index)

        all_model_data = {
            "question": question,
            "answer": answer,
            "other_results": model_evaluations
        }

        return all_model_data


    def answer_question(self, question: str) -> dict:
        """
        This function searches Wikipedia to provide an answer for a given question.
        It returns a dict with two keys: "answer" and "other_results". "other_results"
        is a list but contains entries with an identical structure to "answer". 
        answer_question(question)["answer"]["answer"] contains the answer to the question.

        Parameters
        ----------
        question : str
            A question or query.

        Returns
        -------
        dict
            A dict that contains the query results.
        """
        # Subtract 150 because BERT's tokenizer may return more tokens than my own.
        self.chunk_size = 509 - len(self._get_tokens(question)) - 150

        # Make sure that the query is not too long. Long queries hurt performance.
        question_token_len = len(self._get_tokens(question))
        if question_token_len > 15:
            raise BertTokenSizeOutOfRange(question_token_len)

        # Download all the relevant Wikipedia articles.
        articles = get_articles(question, num_articles_search=self.num_articles_search, characters_per_article=self.characters_per_article)

        # Collect tuples of article chunks: (article_title, article_chunk)
        output = []

        for article_title, article_text in articles:
            chunks = list(self._get_article_chunks(article_text))
            for article_chunk in chunks:
                logging.info("Getting model prediction")
                pred = get_model_predictions(question, article_chunk, self.model_server_address)

                data = {
                        "answer": pred["answer"],
                        "context": article_chunk,
                        "context_article_title": article_title,
                        "start_scores_max": pred["start_scores_max"],
                        "end_scores_max": pred["end_scores_max"],
                        "start_scores": pred["start_scores"],
                        "end_scores": pred["end_scores"]
                        }

                output.append(data)

        return self._decider(output, question)
