from distutils.core import setup
from pathlib import Path

__version__ = "0.0.1"


setup(
    name="embedding-ft",
    version=__version__,
    author="eddie",
    packages=[
        "embedding-ft",
    ],
    install_requires=Path("requirements.txt").read_text().splitlines(),
)
