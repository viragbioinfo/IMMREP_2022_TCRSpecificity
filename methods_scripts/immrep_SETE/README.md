### immrep_SETE.py

Train SETE over the ImmRep2022 training data and test over test data.
Training and testing are done separately for task1 and task2. For task1, a binary
classifier is trained for each epitope separately. For task2, a multiclass classifier
is trained over all epitope-specific TCRs in the training data. <br>
The SETE method only uses the TCR CDR3-beta. Duplicates are removed from each epitope's training set (based on CDR3-beta only).<br><br>
The script is assumed to be placed in the following directories tree:

python_projects/ <br>
&emsp;&emsp;IMMREP_2022_TCRSpecificity/<br>
&emsp;&emsp;&emsp;&emsp;methods_scripts/<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;immrep_SETE/<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;immrep_SETE.py &emsp; <----------<br>
&emsp;&emsp;SETE/<br>
&emsp;&emsp;&emsp;&emsp;SETE.py<br>
&emsp;&emsp;&emsp;&emsp;...<br>
&emsp;&emsp;...<br><br>

Script results will be written to new folders created under the immrep_SETE folder (where the script is located).

Clone the SETE code from
https://github.com/wonanut/SETE
or, if my Git pull request is not accepted yet, please clone repository
https://github.com/liel-cohen/SETE instead. (the original wonanut code will not work with this script!)

Please notice that the cloned SETE folder should be placed in the root folder
(parent folder of the IMMREP_2022_TCRSpecificity folder).
If the SETE folder is located elsewhere, please change the sete_package_folder variable in the
"Imports" part of the script, to the correct SETE folder string.<br>

Good luck! <br>
Liel Cohen-Lavi

