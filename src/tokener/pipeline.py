"""Command-line version of the tokenizer comparison in ``notebooks/``.

Usage:
    med-tokener                                  # full run (notebook-sized slices)
    med-tokener --n-train 1000 --vocab-size 10000  # smaller/faster run
"""

from med_tokener.pipeline import *  # noqa: F403
from med_tokener.pipeline import main

if __name__ == "__main__":
    main()
