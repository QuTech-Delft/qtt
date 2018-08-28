---
title: 'QTT (Quantum Technology Toolbox): a package for tuning and calibration of spin-qubits'
tags:
   - quantum dots
   - spin qubits
   - tuning
   - calibration
authors:
  - name: Pieter Eendebak
    affiliation: "1 2"
  - name: additional authors
    affiliation: "2"
affiliations:
  - name: TNO
    index: 1
  - name: TU Delft
    index: 2
date: 26 June 2018
bibliography: paper.bib
---

# Summary

The Quantum Technology Toolbox (QTT) is a software package containing measurement routines and analysis functions 
for the tuning and calibration of spin-qubits. An introduction to spin-qubits can be found~[@LievenSpins].

The package is organized in the following subpackages:

* Measurements
* Algorithms
* Simulation
* Instrument Drivers
* Gui
* Utilities

The measurements are performed using scanning functions are return a QCoDeS dataset. The input
for analysis functions is the raw data (in the form of a QCoDeS dataset) and a set of parameters
for the algorithm. The result structure from an analysis routine is a dictionary with the key results.

The methods from this paper have been used in the following papers: [@Baart2016], [@Diepen2018], [@Mukhopadhyay2018].

# Acknowledgements

We acknowledge support from the Vandersypen group (the spin-qubit) group in QuTech.

# References
