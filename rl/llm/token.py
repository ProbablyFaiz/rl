from transformers import PreTrainedTokenizer


def token_chunk(
    text: str,
    num_tokens: int,
    tokenizer: PreTrainedTokenizer,
) -> list[str]:
    """Tokenize a long text into chunks of a specified number of tokens."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), num_tokens):
        chunks.append(tokenizer.convert_tokens_to_string(tokens[i : i + num_tokens]))
    return chunks
