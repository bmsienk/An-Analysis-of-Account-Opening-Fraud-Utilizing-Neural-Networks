# An Analysis of Account Opening Fraud Utilizing Neural Networks
### Flatiron School Data Science Capstone Project
**Author: [Brandon Sienkiewicz](https://www.linkedin.com/in/brandon-sienkiewicz-data-science/)**

## I. Project Overview

The rise of technology in the modern era has brought about numerous advances that have helped to improve the life and wellbeing of the general population. This rise in technology has had negative consequences as well. One of such consequences is the increase of fraud reports. Transactional fraud is well monitored and has an abundance of public data sets available, allowing for the analysis and improvement of fraud detection methods utilizing concepts such as machine learning. While there is a vast array of information on transactional fraud data, the data available for account opening fraud is limited. Holding a bank account is essential to survive in the modern world and the issue of account opening fraud is becoming more pressing. FiVerity, a cybersecurity company stated that, "As much as 50% of new U.S. accounts in 2021 were fraudulent" while the U.S. Department of Labor OIG states that account opening fraud constituted roughly $163 billion dollars of total fraud amounts. [<sup>{[1]}](https://www.bankinfosecurity.com/new-fraud-on-block-causes-bank-losses-to-mount-a-18867) As the issue of account opening fraud becomes more prevalent, so will the need to meansures to counteract it. That being said, machine learning provides a strong tool for predeictive modelling for the purposes of combating fraud. My goal for this project was to make predictions for the problem of account opening fraud utilizing deep-learning neural networks.

## II. Data Understanding
 
The data that I utilized for the modelling and analysis was the Bank Account Fraud Dataset Suite [<sup>{[2]}](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=Variant+III.csv), published at the 2022 Conference on Neural Information Processing Systems. The suite of datasets consists of six synthetic account opening fraud data variants based on real-world bank account opening fraud detection data. These variants have differing levels of group disparity and separability for age group, which was split into applicants under the age of 50 and over the age of 50. The base data set is sampling to best represent the original data. Variant I has higher group size disparity than the base data. Variant II has a higher prevalence disparity than the base data. Variant III has better seperability for one of the groups. Variant IV has higher prevalence disparity in the training data. And finally, Variant V has better seperability in the training data for one of the groups. Each data set consists of one million instances, of which 11,029 are denoted as fraudulent. 
 
## III. Data Preparation
 
 

## IV. Data Modelling

## V. Evaluation and Conclusions

## Contact

* **[LinkedIn Profile](https://www.linkedin.com/in/brandon-sienkiewicz-data-science/)**
* **[Email Address](bmsienk@outlook.com)**

## Repository Structure

* [Images](./Images) (Folder containing visualizations for the presentation/readme)
* [Notebooks](./Notebooks) (Folder containing all jupyter notebooks used for this project)
  * [Data Prep](./Notebooks/01-DataPrep) (Data prep and train/validation/test file creation)
  * [Data Modelling (Base)](./Notebooks/02-DataModelling(Base)) (Contains all details for the models used for all data sets)
  * [Data Modelling (Variant I)](./Notebooks/03-DataModelling(VariantI)) (Models from base applied to variant I)
  * [Data Modelling (Variant II)](./Notebooks/04-DataModelling(VariantII)) (Models from base applied to variant II)
  * [Data Modelling (Variant III)](./Notebooks/05-DataModelling(VariantIII)) (Models from base applied to variant III)
  * [Data Modelling (Variant IV)](./Notebooks/06-DataModelling(VariantIV)) (Models from base applied to variant IV)
  * [Data Modelling (Variant V)](./Notebooks/07-DataModelling(VariantV)) (Models from base applied to variant V)
  * [Final Model Results and Analysis](./Notebooks/08-FinalModelResultsAnalysis) (Final Model training, results, and analysis)
 * [.gitignore](./.gitignore)
 * [README](./README.md)
 * [Fuctions Script](./functions.py) (A python script containing useful functions utilized for this project)

## Citing Work

* **Jesus, Sérgio and Pombal, José and Alves, Duarte and Cruz, André and Saleiro, Pedro and Ribeiro, Rita P. and Gama, João and Bizarro, Pedro**. *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*. arXiv (2022). [https://doi.org/10.48550/arxiv.2211.13358](https://doi.org/10.48550/arxiv.2211.13358).
