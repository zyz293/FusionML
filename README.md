# Heterogeneous feature fusion based machine learning on shallow-wide and heterogeneous-sparse industrial dataset
This software is an application of heterogeneous feature fusion based machine learning on shallow-wide and heterogeneous-sparse scientific dataset. The input is a numpy array, containing the images and target values. In this software, a machine learning framework including transfer learning, heterogeneous feature fusion, principal component analysis and gradient boosting is used to solve curse of dimensionality, handle data with missing images, and train predictive models on heterogeneous industrial data. The detailed description about data preprocessing and model can be found in the published paper given below.

## Requirements ##
* Python 3.6.3 
* Numpy 1.18.1 
* Sklearn 0.20.0 
* Keras 2.3.1 
* Pickle 4.0 
* TensorFlow 2.1.0 
* Scipy 1.2.0

## Files ##
1. `model_training.py`: The script applies the proposed method to train a prediction model on shallow-wide and heterogeneous-sparse scientific dataset. The results will be saved in `results.csv` file.
2. `data` folder: This folder has `data.pkl`, which is the example dataset, because `Toyota dataset` is not public available due to confidential reasons. 


## How to run it
1. Run commend below, which uses proposed method to train the model and save results in `results.csv` file.
   ```
   python model_training.py
   ```

## Acknowledgement
This work was supported in part by Toyota Motor Corporation and NIST CHiMaD (70NANB19H005).

## Related Publications ##
Zijiang Yang, Tetsushi Watari, Daisuke Ichigozaki, Akita Mitsutoshi, Hiroaki Takahashi, Yoshinori Suga, Wei-keng Liao, Alok Choudhary, Ankit Agrawal. "Heterogeneous feature fusion based machine learning on shallow-wide and heterogeneous-sparse industrial datasets". Accepted by 1st International Workshop on Industrial Machine Learning @ ICPR 2020.

## Contact
Zijiang Yang <zyz293@ece.northwestern.edu>; Ankit Agrawal <ankitag@ece.northwestern.edu>
