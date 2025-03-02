# Speech omics representation

This is implementation of our manuscript "A Neuromorphic Speech Computing Framework for Machine Learning-based Diagnosis and Prognosis in Oral and Oropharyngeal Cancers"


### 1_Data_prepare
This step is implemented in Praat software (https://www.fon.hum.uva.nl/praat/). The raw audio files (could be downloaded at https://pan.baidu.com/s/1kB-lwtY16M43mzdAIlaMoQ?pwd=25m4) would be preprocessed including denoising via spectral subtraction method and amplitude normalization. Then pertinent feature arrays(/F1a/, /F2a/, /F3a/, /B1a/, /B2a/, /B3a/, /F1i/, /F2i/, /F3i/, /B1i/, /B2i/, /B3i/, /F1u/, /F2u/, /F3u/, /B1u/, /B2u/, /B3u/) would be extracted from these preprocessed audio files. 
Input: raw audio file with extension of ".wav"
Output: table file containing original frame-wise feature arrays with extention of ".Table"

### 2_SOR
Code for speech omics representation. 
Input: individual-level table files containing original frame-wise feature arrays. They can be downloaded directly at https://pan.baidu.com/s/1qTxUGQZ1USD-ABgWZLjSqg?pwd=ma5e. In this way, you can skip 1_Data_prepare.
Output: group-level table sheets containing speech omics features for all speakers. 
Code example: speech_omics_representation.ipynb

### 3_omics_attri
Code for the characterization of omics attribute. Weight graphs are used to visualize the spatial layout among speech omics features. Redundancy atrributes are analyzed with regard to specific biologic profiles.
Modularity attributes are analyzed using Gaussian kernel canonical correlation analysis (CCA), resulting derived CCA-based metrics. In biological characterization, effect sizes are quantified using Cohen's d values for categorical variables, and Pearson or Spearman correlation coefficients for continuous variables. 
Code example: weight_graph.ipynb, Modularity_CCA_metrics.ipynb, Redundancy.ipynb

### 4_machine_learning
Code for machine learning in diagnostic tasks. Before analysis, please download Task.xlsx at https://pan.baidu.com/s/1eLur_ntPNCFbvo9f87-puQ?pwd=rs0d, and put it into 0.basic folder. 
Code example: machine_learning.ipynb

### 5_prognostic_cluster
Code for prognostic analysis, including manifold learning, partitioning around the medoid clustering, and log-rank analyses with accelerated failure time (AFT) models. 
Code example: prognostic_analysis.ipynb

### 6_ex_val
Code for external validation. 
Code example: external_validation.ipynb

