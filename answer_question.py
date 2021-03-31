from typing import List

from document_retrieval import get_articles
from bert_model import get_model_predictions # TODO: if model_server="something", this shouldnt be imported


class Answerer:
    def __init__(self, num_articles=5, characters_per_article=2500, chunk_size=300, model_server=False):
        self.num_articles = num_articles
        self.characters_per_article = characters_per_article
        self.chunk_size = chunk_size
        self.model_server = model_server


    def _get_article_chunks(self, article: str) -> List[str]:
        """
        This function divides an article into smaller chunks. BERT has an upper limit for
        the number of tokens it can accept, so the chunk size must be accordingly small.
        BERT can accept 509 tokens (512 minus the three special tokens). So the length of
        the question + answer must be <= 509.
        TODO: currently this function chunks based on whitespace tokenization, NOT Bert's
        own tokenizer - replace with BERT.
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
        tokens = article.split(" ")
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
        assert len(question.split(" ")) < 15

        # Download all the relevant Wikipedia articles.
        articles = get_articles(question, num_articles=self.num_articles, characters_per_article=self.characters_per_article)

        # Collect tuples of article chunks: (article_title, article_chunk)
        output = []

        for article_title, article_text in articles:
            chunks = list(self._get_article_chunks(article_text))
            for article_chunk in chunks:
                pred = get_model_predictions(question, article_chunk)

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


if __name__ == "__main__":

    QUESTION = "how many provinces are in Canada?"
    answerer = Answerer()

    print(QUESTION)
    print("----------")
    ans = answerer.answer_question(QUESTION)
    print(ans["answer"])

