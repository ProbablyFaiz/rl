"""LLM-based detectors output a "raw_passage" string that is drawn from
the OCR text of the image. Because the LLM sometimes outputs a passage
that slightly differs from the OCR text, we need to fuzzy match the
passage to find the character span of the passage in the OCR text.
"""

import dataclasses
import difflib
import re

import unidecode

from rl.utils import LOGGER

_TRIM_EDGE_NEWLINES_LIMIT = 20


@dataclasses.dataclass
class DocumentFragment:
    start: int
    end: int
    text: str


# TODO: This is very slow, and should be optimized.
def get_best_ngram_match(
    passage: str,
    document: str,
    *,
    ngram_size: int = 3,
    max_passage_size: int = 512,
    stride_size: int = None,
    line_align: bool = True,
    sentence_align: bool = False,
) -> tuple[int, int] | None:
    if line_align and sentence_align:
        LOGGER.warning(
            "Both line_align and sentence_align are set to True. "
            "This is not recommended, and you have only yourself to "
            "blame if strange results ensue."
        )

    if not passage or not document:
        return None
    # if there's an exact match, return it
    if passage in document:
        start = document.find(passage)
        return start, start + len(passage)

    passage = _clean_text(passage[:max_passage_size])
    if not stride_size:
        stride_size = len(passage) // 4

    frag_chunk_size = len(passage) + stride_size
    doc_fragments = _fragment_document(
        document, chunk_size=frag_chunk_size, stride=stride_size
    )
    if not doc_fragments:
        return None
    matched_fragment = max(
        doc_fragments,
        key=lambda fr: _get_ngram_text_similarity(passage, fr.text, n=ngram_size),
    )

    # now, let's do this again, striding by word in just the area
    #  surrounding the matched fragment
    frag_chunk_size = len(passage) + 10
    new_start = max(matched_fragment.start - stride_size, 0)
    new_end = min(matched_fragment.end + stride_size, len(document))
    doc_fragments = _fragment_document(
        document[new_start:new_end],
        chunk_size=frag_chunk_size,
        stride="word",
    )
    if not doc_fragments:
        return None
    new_fragment = max(
        doc_fragments,
        key=lambda fr: _get_ngram_text_similarity(passage, fr.text, n=ngram_size),
    )
    new_fragment.start += new_start
    new_fragment.end += new_start

    start, end = new_fragment.start, new_fragment.end
    if line_align:
        # If there is a newline in the first or last 10 characters of the span,
        #  let's chop it off. We do this because it mitigates slight imprecision
        #  in the resultant span, and avoids leading to a bounding box that
        #  sweeps in extra lines.
        trim_limit = _TRIM_EDGE_NEWLINES_LIMIT
        if "\n" in document[start : start + trim_limit]:
            # We use rfind here because we want to get the last newline before the
            #  trim limit, not the first newline after the trim limit.
            # start = document[start : start + trim_limit].rfind("\n") + start + 1
            start = document.rfind("\n", start, start + trim_limit) + 1
        if "\n" in document[end - trim_limit : end]:
            # First, get the index of the closest newline to the start of this range:
            closest_newline = document.find("\n", end - trim_limit, end)
            next_newline = document.find("\n", closest_newline + 1)
            if next_newline == -1:
                next_newline = len(document)
            # If there is a short line at the end, it's probably the end of the span. let's keep it.
            if 1 <= next_newline - closest_newline <= trim_limit:
                end = next_newline
            else:
                end = document[end - trim_limit : end].find("\n") + (end - trim_limit)
    if sentence_align:
        from legal_segmenter.segmenter import Segmenter

        segmenter = Segmenter()
        sentences = [
            sent
            for para in segmenter.segment(
                document[max(0, start - 512) : min(len(document), end + 512)],
                include_metadata=True,
            )
            for sent in para["sentences"]
        ]
        start = min(
            (s["start"] for s in sentences if abs(s["start"] - start) < 32),
            key=lambda ss: abs(ss - start),
            default=start,
        )
        end = max(
            (
                s["end"]
                for s in sentences
                if abs(s["end"] - end) < 32 and s["end"] > start
            ),
            key=lambda se: abs(se - end),
            default=end,
        )

    return start, end


def _fragment_document(
    doc: str, chunk_size: int, stride: int | str
) -> list[DocumentFragment]:
    """Chunks the document into DocumentChunks of size chunk_size, with stride stride.

    Args:
        doc: The document to chunk.
        chunk_size: The size of each chunk.
        stride: The stride between each chunk.

    Returns:
        A list of DocumentChunks.
    """
    if stride == "word":
        start_indices = [m.start() for m in re.finditer(r"\w+", doc)]
    else:
        start_indices = range(0, len(doc), stride)
    end_indices = [i + chunk_size for i in start_indices if i + chunk_size <= len(doc)]
    chunks = []
    for i, j in zip(start_indices, end_indices, strict=False):
        chunks.append(
            DocumentFragment(
                start=i,
                end=j,
                text=_clean_text(doc[i:j]),
            )
        )
    return chunks


def _clean_text(text: str) -> str:
    # coerce to ascii
    text = unidecode.unidecode(text)
    # remove all whitespace sequences with a single space
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^A-Za-z0-9 \n]+", "", text).lower()


def _get_ngrams(text: str, n: int = 3) -> list[str]:
    """Returns the ngrams of the given text.

    Args:
        text: The text to get the ngrams of.
        n: The number of characters in each ngram.

    Returns:
        A list of ngrams.
    """
    if not text:
        return []
    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _get_ngram_text_similarity(text1: str, text2: str, n: int = 3) -> float:
    return difflib.SequenceMatcher(
        None, _get_ngrams(text1, n), _get_ngrams(text2, n)
    ).ratio()
