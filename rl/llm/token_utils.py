import more_itertools as mit
from transformers import PreTrainedTokenizer


def token_chunk(
    text: str,
    *,
    chunk_size: int,
    stride: int | None = None,
    tokenizer: PreTrainedTokenizer,
) -> list[str]:
    """Tokenize a long text into chunks of a specified number of tokens."""
    if stride is None:
        stride = chunk_size

    tokens = tokenizer.tokenize(text)
    chunks = []
    for chunk in mit.windowed(tokens, n=chunk_size, step=stride):
        chunks.append(tokenizer.convert_tokens_to_string(list(chunk)))
    return chunks
