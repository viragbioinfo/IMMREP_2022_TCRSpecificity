### Test set
This directory contains the test set. The evaluation of different models is done in two modes:

1. Epitope-specific mode: There are epitiope specific files available here. Each file contains "x" number of positives and "5x" number of negatives. The objective is to evaluate how well each model (trained for the specific-epitope) separates the positives from the negatives.

2. Global mode: This set (testSet_Global.txt) contains only the positive TCRs for the 17 different epitopes. The objective is to evaluate how well the different models predict the correct epitope for a TCR under consideration.
