# IMMREP_2022_TCRSpecificity

This repository contains scripts and trainining and test data for Working Group 6 (Can TCR specificities be predicted?) of the "Dynamics of Immune Repertoires: Exploration and Translation" workshop held in Dresden from 7-30 July 2022


## Training data: 23/07/2022

The data contains TCRa and TCRb chain CDR3 + VDJ information obtained by 10x genomics sequencing or other single-cell sequencing protocols. Note that only TCRs with paired information, i.e. from both, alpha and beta chains, have been retained.
Data with peptide specificity information originates from VDJdb (https://vdjdb.cdr3.net/). There are TCRs with specificity to 17 peptides (from different studies that have been collated by VDJdb). Data without peptide information (control) is an unpublished data set from A. Eugster containing sequences from CD8+CD96+ from 11 individuals.  

The data has been assembled in the following way: 

**Positives:** 
Every epitope, (for example ATDALMTGF) has x TCRs (for example x=132) with peptide information, these are labelled as the “positive” samples for every epitope.
**Negatives:** 
As “negative” for each epitope, we have randomly sampled and combined TCRs from all other 16 epitopes, to obtain a 3 fold number of the positive TCRs (for ATDALMTGF, this would be 396 TCRs). In addition, we have randomly sampled TCRs from the control dataset to obtain the twice the number of positive TCRs (for ATDALMTGF, this would be 264 TCRs). In total for this epitope there are therefore 132 positives and 660 negatives.

Training and test data
Before merging the positives and negatives together for each epitope, the data was split into training and test set in the ratio of **80:20.**

The dataset for training for each epitope is present in the training_data directory in this repository. Every file contains the usual information associated with a TCR. This includes the nucleotide sequence of the full length TCR, the amino acid sequence of the full length TCR, the CDR3 region and lastly, the information about different genes that are being used by a specific TCR. The full-length TCRs were obtained using the tool called Stitchr.(https://github.com/JamieHeather/stitchr).
In addition, the full length TCR sequence was further decomposed to obtain the CDR1 and the CDR2 region. This was done using the package Anarci (https://github.com/oxpig/ANARCI). This information shows up in the file for training data for every epitope under the headers ** A1, A2 and A3 **(CDR1, CDR2 and CDR3 regions of the alpha chain) and ** B1, B2 and B3 **(DR1, CDR2 and CDR3 regions of the beta chain).
Many thanks to Alessandro Montemurro (from the group of Morten Neilsen) for providing the wrapper script for this functionality.

The training datatset for every epitope contains a value (either 1 or -1) under the column **Label**. This value indicates if a particular TCR is a positive sample or a negative sample.

**Note:** Currently, the training data for each epitope contains “x” number of positive samples and “5x” number of negative samples. Users may need to downsample the count of negative examples for training their models.

### Further Information regarding the control data set: 
The complete set of TCRs from control is further provided so that you can modify the training and test data as you please. For example, by modifying the proportion of controls included, and likewise increase or reduce the proportion of swapped negatives. 

## Test data: Will be released during the workshop
