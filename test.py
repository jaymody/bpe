from bpe import BPETokenizer
from original import Encoder


def load_original_tokenizer(tokenizers_dir):
    import json
    import os

    with open(os.path.join(tokenizers_dir, "encoder.json"), "r") as f:
        encoder = json.load(f)
    with open(os.path.join(tokenizers_dir, "vocab.bpe"), "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )


def test(text_file, tokenizers_dir):
    with open(text_file) as f:
        text = f.read()

    new_tokenizer = BPETokenizer.load(tokenizers_dir)
    old_tokenizer = load_original_tokenizer(tokenizers_dir)

    assert text == new_tokenizer.decode(new_tokenizer.encode(text))
    print("✅ test passed (encode -> decode recovers input text)")

    assert new_tokenizer.encode(text) == old_tokenizer.encode(text)
    print("✅ test passed (gives same output as original implementation)")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=str)
    parser.add_argument("--tokenizers_dir", type=str, default="tokenizer/")

    args = parser.parse_args()
    test(args.text_file, args.tokenizers_dir)


if __name__ == "__main__":
    main()
