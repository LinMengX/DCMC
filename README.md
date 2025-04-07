# Decoupled contrastive multi-view clustering with adaptive false negative elimination for cancer subtyping
![Overview of DCMC](DCMC_main/figure/overview_DCMC.png)
## Requirements

pytorch>=1.13.0 

numpy>=1.23.4

scikit_learn>=1.4.2

scipy>=1.10.1

All the specific packages required can be found in requirements.txt

## Datasets

The 10 cancer multi-omics data and patient clinical data used in this article can be obtained through the TCGA public platform https://portal.gdc.cancer.gov. All the multi-omics data, survival data, and clinical data of patients in this experiment are taken from http://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html. 

## Training

The hyper-parameters, the training options are defined in the configure file.

~~~bash
main_train.py --config_file=config/Scene15.yaml
~~~

~~~bash
main_train.py --config_file=config/Caltech101.yaml
~~~

## Reference


