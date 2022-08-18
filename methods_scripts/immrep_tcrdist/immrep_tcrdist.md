### immrep_tcrdist.py

Train a k-nearest neighbours (KNN) classifier using TCRdist distances
over the ImmRep2022 training data and test over test data.
Training and testing are done separately for task1 and task2.
The task2 folder has results from 2 types of models:<br>
* The files that end with "binary" contain predictions from each epitope's
binary model (trained for task 1).<br>
* The files that end with "multi" are created by getting predictions from a multiclass model
trained on only epitope-specific TCRs from the training data files.<br><br>

The script is assumed to be placed in the following directories tree:

python_projects/ <br>
&emsp;&emsp;IMMREP_2022_TCRSpecificity/<br>
&emsp;&emsp;&emsp;&emsp;methods_scripts/<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;immrep_tcrdist/<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;immrep_tcrdist.py &emsp; <----------<br>


Script results will be written to new folders created under the immrep_tcrdist folder (where the script is located).

tcrdist3 package must be installed in the environment. You can use manual installation or a docker container.
Full installation instructions are available at:
https://tcrdist3.readthedocs.io/en/latest/index.html

*If using the tcrdist3 docker container, the sklearn package should be pip 
installed in it as well.<br>
**If using a different location for the script, just make sure to change paths
to the train and test data folders under the Params section:
paths.input_training_data_folder, paths.input_test_data_folder<br>
(or change paths.folder)<br><br>

Good luck!<br>
Liel Cohen-Lavi