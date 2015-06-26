pyconsensus
===========

.. image:: https://badge.fury.io/py/pyconsensus.svg
    :target: http://badge.fury.io/py/pyconsensus

pyconsensus is a standalone Python implementation of Augur's consensus mechanism.  For details, please see the `Augur whitepaper`_.

Installation
^^^^^^^^^^^^

The easiest way to install pyconsensus is to use pip::

    $ pip install pyconsensus

Usage
^^^^^

To use pyconsensus, import the Oracle class:

.. code-block:: python

    from pyconsensus import Oracle

    # Example report matrix:
    #   - each row represents a reporter
    #   - each column represents an event in a prediction market
    my_reports = [[0.2, 0.7,  1,  1],
                  [0.3, 0.5,  1,  1],
                  [0.1, 0.7,  1,  1],
                  [0.5, 0.7,  2,  1],
                  [0.1, 0.2,  2,  2],
                  [0.1, 0.2,  2,  2]]
    reputation = [1, 2, 10, 9, 4, 2]
    my_event_bounds = [
        {"scaled": True, "min": 0.1, "max": 0.5},
        {"scaled": True, "min": 0.2, "max": 0.7},
        {"scaled": False, "min":  1, "max": 2},
        {"scaled": False, "min":  1, "max": 2},
    ]

    oracle = Oracle(reports=my_reports,
                    reputation=reputation,
                    event_bounds=my_event_bounds)
    oracle.consensus()

Tests
^^^^^

Unit tests are in the test/ directory.

pyconsensus is used in conjunction with the `Augur Simulator`_ to carry out randomized numerical (Monte Carlo) consensus tests.  See the Simulator_ repository for details.

.. _Augur whitepaper: http://augur.link/augur.pdf
.. _Augur Simulator: https://github.com/AugurProject/Simulator.jl
.. _Simulator: https://github.com/AugurProject/Simulator.jl
