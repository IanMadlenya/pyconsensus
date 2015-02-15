#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="pyconsensus",
    version="0.4.1",
    description="Standalone implementation of Augur's consensus mechanism",
    author="Jack Peterson and Paul Sztorc",
    author_email="<jack@tinybike.net>",
    maintainer="Jack Peterson",
    maintainer_email="<jack@tinybike.net>",
    license="GPL",
    url="https://github.com/tensorjack/pyconsensus",
    download_url = "https://github.com/tensorjack/pyconsensus/tarball/0.4.1",
    packages=["pyconsensus"],
    install_requires=["numpy", "pandas", "six", "weightedstats", "scipy", "scikit-learn"],
    keywords = ["consensus", "prediction market", "PM", "truthcoin", "oracle", "PCA"]
)
