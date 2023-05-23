import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def get_token_length(sentences):
    return [len(enc.encode(s)) for s in sentences]
