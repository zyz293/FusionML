# Heterogeneous feature fusion based machine learning on shallow-wide and heterogeneous-sparse scientific dataset
This software is an application of heterogeneous feature fusion based machine learning on shallow-wide and heterogeneous-sparse scientific dataset. The efficacy of the proposed approach is tested on four datasets, including two artificial dataset and two real world dataset. 

To use this software, what the algorithm requires as input are a numpy array. In this software, a machine learning framework including transfer learning, heterogeneous feature fusion, principal component analysis and gradient boosting is used to solve curse of dimensionality, handle data with missing images, and train predictive models on heterogeneous scientific data. The detailed drscription about data preprocessing and model can be found in the published paper given below.

## Requirements ##
* Python 3.6.3 
* Numpy 1.18.1 
* Sklearn 0.20.0 
* Keras 2.3.1 
* Pickle 4.0 
* TensorFlow 2.1.0 
* Scipy 1.2.0

## Files ##
1. `model_training.py`: The script applies the proposed method to train a prediction model on shallow-wide and heterogeneous-sparse scientific dataset. The results will be saved in `results.csv` file. We train the model on artificial dataset No.1 as an example.
2. `data_generation.pkl`: The script is used to generate dataset, and we use the generation of artificial dataset No.1 as an example. The generated data will be saved in `data.pkl` file
4. `data` folder: This folder includes four datasets used in this work. `data.pkl` is the artificial dataset No.1, and `data_RASTRIGIN.pkl` is the artificial dataset No.2. `XRD dataset` is available on request, and `Toyota dataset` is not public available due to confidential reasons. 


## How to run it
1. Run commend below, which generates the dataset and save it in `data.pkl` file.
   ```
   python data_generation.py
   ```
1. Run commend below, which uses proposed method to train the model and save results in `results.csv` file.
   ```
   python model_training.py
   ```

## Acknowledgement


## Related Publications ##
Heterogeneous feature fusion based machine learning on shallow-wide and heterogeneous-sparse scientific dataset (in preparation)

## Contact
Zijiang Yang <zyz293@ece.northwestern.edu>; Ankit Agrawal <ankitag@ece.northwestern.edu>
