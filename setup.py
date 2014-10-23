#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="pyconsensus",
    version="0.2",
    description="Consensus mechanism for Truthcoin",
    author="Jack Peterson and Paul Sztorc",
    author_email="<jack@tinybike.net>",
    maintainer="Jack Peterson",
    maintainer_email="<jack@tinybike.net>",
    license="GPL",
    url="https://github.com/tensorjack/pyconsensus",
    download_url = "https://github.com/tensorjack/pyconsensus/tarball/0.2",
    packages=["pyconsensus"],
    install_requires=["numpy", "six", "weightedstats"],
    keywords = ["consensus", "prediction market", "PM", "truthcoin", "oracle"]
)
