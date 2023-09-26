A simplified implementation of OpenAI's BPE encoder for GPT-2.

The original implementation can be found in `original.py`, which was copied from [here](https://github.com/openai/gpt-2/blob/master/src/encoder.py).

My re-implementation can be found in `bpe.py`. I simplified a lot of things, added type hints, and refactored everything to be functional (I use recursion for merging the pairs). This implementation is probably slower than the original.

You can test that this implementation gives identical outputs to the original when encoding `some_text_file.txt` via:

```shell
$ python test.py some_text_file.txt
✅ test passed (encode -> decode recovers input text)
✅ test passed (gives same output as original implementation)
```

Note, you'll need to install `regex`:

```shell
$ pip install regex
```

Tested with `Python 3.9.6`.
