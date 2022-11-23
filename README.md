# MMDRP, Multi-Modal Drug Response Prediction
Preprocessing, training and evaluation code for MMDRP models.


## Preprocessing
Training data was obtained from the following:
- CTRPv2 was obtained and processed using the PharmacoGx BioConductor package (https://bioconductor.org/packages/release/bioc/html/PharmacoGx.html)
  * Please refer to the `R/01_Dose-Response_Data_Preparation.R` file for details.
- DepMap Portal (https://depmap.org/portal/) for cell line profiling data.
  * 20Q2 for Protein Quantification data (lastest) and 21Q2 for mutational, gene expression, CNV, miRNA, metabolomics, histone, and RPPA data.
  * Please refer to the `R/02_Omic_Data_Preparation.R` file for details.
## Training
Training was done in Python using the Pytorch framework. `.py` files are available in the `src` folder.  
`drp_full_model.py`is the main file used for training which can be run as a commandline program. Please refer to this file for the list of input arguments and their defaults.  

## Evaluation
Evaluation was performed using multiple cross-validation schemes. The predictions from the validation sets were then aggregated for each model, and further analyzed and compared in the `05_All_Comparison_Plots.R` file.
