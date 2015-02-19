Python Implementation of Word Alignment for Machine Translation.
Run as ‘python <filename>.py > <alignment_file>’
Options:
-n num_lines		take fewer lines from input files
-d <data_file>		take files other than default, e.g.: sample

ibm1.py: Implements IBM Model 1
ibm2.py: Implements IBM Model 2
hmm.py:  Implements HMM Model for Word Alignment
fastAlign.py: Implements fast align method, but with fixed lambda parameter
alignIntersect.py: Implements intersection-based alignment using Dice’s coefficient
modelAgreement.py: Implements intersection-based alignment for IBM Model2

Files for corresponding names with .a extension are alignments for first 1000 sentences obtained from these models trained on full hansards data for English and French. They can be run with score-alignments for AER evaluation.

t_ibm1 has the translation probabilities from IBM1 that are fed to the advanced models.
It has been compressed since it is a large file and should be uncompressed for use in the advanced models.

sample.e and sample.f are small 3-line corpus for quick-check of model and should result in: 
0-0 1-1
0-1 1-0
0-0 1-1

