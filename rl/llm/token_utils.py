from transformers import PreTrainedTokenizerFast


def token_chunk(
    text: str,
    *,
    chunk_size: int,
    stride: int | None = None,
    tokenizer: PreTrainedTokenizerFast,
) -> list[str]:
    """Tokenize a long text into chunks of a specified number of tokens."""
    if stride is None:
        stride = chunk_size

    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + chunk_size]
        chunks.append(tokenizer.convert_tokens_to_string(list(chunk)))
    return chunks
