pyconsensus
===========

.. image:: https://travis-ci.org/AugurProject/pyconsensus.svg?branch=master
    :target: https://travis-ci.org/AugurProject/pyconsensus

.. image:: https://coveralls.io/repos/AugurProject/pyconsensus/badge.png
  :target: https://coveralls.io/r/AugurProject/pyconsensus

.. image:: https://badge.fury.io/py/pyconsensus.svg
    :target: http://badge.fury.io/py/pyconsensus

pyconsensus is a Python implementation of the Augur/Truthcoin consensus mechanism, originally invented by Paul Sztorc: https://github.com/psztorc/Truthcoin.

Donations: 14sqtQRWuWqa7SCtS1iSjt1FexSxfwnw7G

Installation
^^^^^^^^^^^^

The easiest way to install pyconsensus is using pip::

    $ pip install pyconsensus

To upgrade your pyconsensus installation, use::

    $ pip install --upgrade --no-deps pyconsensus

Usage
^^^^^

To use pyconsensus, import the Oracle class::

    from pyconsensus import Oracle

    # Example vote matrix:
    #   - each row represents a voter
    #   - each column represents a decision in a prediction market
    my_votes = [[1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1]]
    my_decision_bounds = [
        {"scaled": True, "min": 0.1, "max": 0.5},
        {"scaled": True, "min": 0.2, "max": 0.7},
        {"scaled": False, "min": 0, "max": 1},
        {"scaled": False, "min": 0, "max": 1},
    ]

    oracle = Oracle(votes=my_votes, decision_bounds=my_decision_bounds)
    oracle.consensus()

Tests
^^^^^

Unit tests are in the test/ directory.
