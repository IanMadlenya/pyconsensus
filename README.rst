pyconsensus
===========

.. image:: https://travis-ci.org/AugurProject/pyconsensus.svg?branch=master
    :target: https://travis-ci.org/AugurProject/pyconsensus

.. image:: https://coveralls.io/repos/AugurProject/pyconsensus/badge.png
  :target: https://coveralls.io/r/AugurProject/pyconsensus

.. image:: https://badge.fury.io/py/pyconsensus.svg
    :target: http://badge.fury.io/py/pyconsensus

Python implementation of the Augur consensus mechanism, which is an improved version of Sztorc consensus (https://github.com/psztorc/Truthcoin).  Mathematical details can be found in Appendix A of the Augur whitepaper at http://augur.link/augur.pdf.

Donations: 14sqtQRWuWqa7SCtS1iSjt1FexSxfwnw7G

Installation
^^^^^^^^^^^^

The easiest way to install pyconsensus is using pip::

    $ pip install pyconsensus

Usage
^^^^^

To use pyconsensus, import the Oracle class:

.. code-block:: python

    from pyconsensus import Oracle

    # Example report matrix:
    #   - each row represents a reporter
    #   - each column represents an event in a prediction market
    my_reports = [[ 0.2,  0.7,  -1,  -1 ],
                  [ 0.3,  0.5,  -1,  -1 ],
                  [ 0.1,  0.7,  -1,  -1 ],
                  [ 0.5,  0.7,   1,  -1 ],
                  [ 0.1,  0.2,   1,   1 ],
                  [ 0.1,  0.2,   1,   1 ]]
    my_event_bounds = [
        {"scaled": True, "min": 0.1, "max": 0.5},
        {"scaled": True, "min": 0.2, "max": 0.7},
        {"scaled": False, "min": -1, "max": 1},
        {"scaled": False, "min": -1, "max": 1},
    ]

    oracle = Oracle(reports=my_reports, event_bounds=my_event_bounds)
    oracle.consensus()

Tests
^^^^^

Unit tests are in the test/ directory.
