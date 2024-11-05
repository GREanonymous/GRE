from __future__ import annotations

from typing import Any, List

from langchain.text_splitter import TextSplitter


class SpacyTextSplitter(TextSplitter):
    """Splitting text using Spacy package.

    Per default, Spacy's `en_core_web_lg` model is used and
    its default max_length is 1000000 (it is the length of maximum character
    this model takes which can be increased for large files). For a faster, but
    potentially less accurate splitting, you can use `pipeline='sentencizer'`.
    """

    def __init__(
            self,
            separator: str = " ",
            pipeline: str = "en_core_web_lg",
            chunk_size: int = 250,
            chunk_overlap: int = 50,
            min_chunk_size: int = 64,
            max_length: int = 1_000_000,
            **kwargs: Any,
    ) -> None:
        """Initialize the spacy text splitter."""
        super().__init__(**kwargs)
        self._tokenizer = _make_spacy_pipeline_for_splitting(
            pipeline, max_length=max_length
        )
        self._separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        doc = self._tokenizer(text)
        sentences = [sent.text for sent in doc.sents]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length + len(current_chunk) - 1 > self.chunk_size:
                if current_chunk:
                    if len(self._separator.join(current_chunk)) >= self.min_chunk_size:
                        chunks.append(self._separator.join(current_chunk))
                    else:
                        if chunks:
                            chunks[-1] += self._separator + self._separator.join(current_chunk)
                        else:
                            chunks.append(self._separator.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + (1 if current_chunk else 0)

        if current_chunk:
            if len(self._separator.join(current_chunk)) >= self.min_chunk_size:
                chunks.append(self._separator.join(current_chunk))
            else:
                if chunks:
                    chunks[-1] += self._separator + self._separator.join(current_chunk)
                else:
                    chunks.append(self._separator.join(current_chunk))

        return chunks


def _make_spacy_pipeline_for_splitting(
        pipeline: str, *, max_length: int = 1_000_000
) -> Any:  # avoid importing spacy
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "Spacy is not installed, please install it with `pip install spacy`."
        )
    if pipeline == "sentencizer":
        from spacy.lang.en import English

        sentencizer: Any = English()
        sentencizer.add_pipe("sentencizer")
    else:
        sentencizer = spacy.load(pipeline, exclude=["ner", "tagger"])
        sentencizer.max_length = max_length
    return sentencizer
