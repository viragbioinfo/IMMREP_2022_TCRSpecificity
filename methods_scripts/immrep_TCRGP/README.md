### immrep_TCRGP.py

Train TCRGP over the ImmRep2022 training data, then test over test data 
for task1 and task2. A binary classifier is trained for each epitope separately. 
For task2, the binary models were used to predict the score for each epitope separately.
For each TCR, the epitopes were then ranked by their predicted scores, and the epitope 
that got the highest score was chosen as the predicted epitope.<br>

The script is assumed to be placed in the following directories tree:

python_projects/ <br>
&emsp;&emsp;IMMREP_2022_TCRSpecificity/<br>
&emsp;&emsp;&emsp;&emsp;methods_scripts/<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;immrep_TCRGP/<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;immrep_TCRGP.py &emsp; <----------<br>
&emsp;&emsp;TCRGP/<br>
&emsp;&emsp;&emsp;&emsp;tcrgp.py<br>
&emsp;&emsp;&emsp;&emsp;...<br>
&emsp;&emsp;...<br><br>

Script results will be written to the immrep_TCRGP folder (where the script is located).

Clone the TCRGP code from
https://github.com/emmijokinen/TCRGP
or, if my git pull request is not accepted yet, clone repository
https://github.com/liel-cohen/TCRGP instead. (the original emmijokinen code will not work with this script!)

Please notice that the cloned TCRGP folder should be placed in the root folder
(parent folder of the IMMREP_2022_TCRSpecificity folder).
If the TCRGP folder is located elsewhere, please change the tcrgp_package_folder variable in the
"Imports" part of the script, to the correct TCRGP folder string.<br><br>

Good luck!<br>
Liel Cohen-Lavi