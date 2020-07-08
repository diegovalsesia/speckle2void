# [Speckle2Void: Deep Self-Supervised SAR Despeckling with Blind-Spot Convolutional NeuralNetworks](https://arxiv.org/abs/2007.02075)

Speckle2Void is a self-supervised Bayesian despeckling framework that enables direct training on real SAR images. This method bypasses the problem of training a CNN on synthetically-speckled optical images, thus avoiding any domain gap and enabling  learning of features from real SAR images. 

This repository contains python/tensorflow implementation of Speckle2Void, trained and tested on the TerraSAR-X dataset provided by ESA [archive](https://tpm-ds.eo.esa.int/oads/access/collection/TerraSAR-X).


BibTex reference:
```
@ARTICLE{2020arXiv200702075B,
       author = {{Bordone Molini}, Andrea and {Valsesia}, Diego and {Fracastoro}, Giulia and
         {Magli}, Enrico},
        title = "{Speckle2Void: Deep Self-Supervised SAR Despeckling with Blind-Spot Convolutional Neural Networks}",
      journal = {arXiv e-prints},
     keywords = {Electrical Engineering and Systems Science - Image and Video Processing, Computer Science - Computer Vision and Pattern Recognition},
         year = 2020,
        month = jul,
          eid = {arXiv:2007.02075},
        pages = {arXiv:2007.02075},
archivePrefix = {arXiv},
       eprint = {2007.02075},
 primaryClass = {eess.IV}
}
```

#### Setup to get started
Make sure you have Python3 and all the required python packages installed:
```
pip install -r requirements.txt
```


#### Get TerraSAR-X data through ESA EO products online search service and build the dataset.
- Download the TerraSAR-X products from the [ESA EO products online search service](https://tpm-ds.eo.esa.int/oads/access/collection/TerraSAR-X)
- Pre-process the dataset through the speckle decorrelator explained in the paper: [Blind speckle decorrelation for SAR image despeckling](https://ieeexplore.ieee.org/document/6487399). The blind-spot networks work properly if the noise is spatially decorrelated. The SAR imaging system correlates the speckle noise in SAR images during acquisition. A speckle decorrelator is needed before performing despeckling.
    * Convert the downloaded TerraSAR-X SLC products into _.mat_ files as complex SAR images. We need complex SAR data to run the speckle decorrelation procedure. It is advised to save in each _.mat_ file a complex SAR image of size 10000x10000 maximum for performance reasons.
    * The _decorrelator.m_ script takes a _.mat_ file as input and estimates the transfer function of the SAR acquisition and focusing system, inverts it and applies it to the complex SAR image. This procedure estimates the complex backscatter coefficients, representing the target scene before going through the acquistion chain. 
      * Required input parameters:
          **input_file**: input .mat file.
          **output_file**: output .mat file representing the decorrelated complex SAR image.
          **cutoff frequencies f_x and f_y**: the cutoff frequencies along each spatial frequency are either supposed to be known   from the technical specifications of the SAR system or manually estimated from the inspection of the average periodograms. Run inspect_periodograms to visualize the periodograms of the original SAR data and choose the cutoff frequencies for both the range and azimuth directions.
          **frequency shifts m_x and m_y**: sometimes the periodograms of the original SAR data are affected by a frequency shift that has been compensated before fitting. Run inspect_periodograms to visualize the periodograms of the original SAR data   and manually choose the recovery frequency shifts for both the range and azimuth directions.
          **cf**: Real SAR images usually contain point targets, which are due to man-made features or edges. Such strong scatterers must be generally preserved because they show a high level of reflectivity with no speckle noise. They have to be detected and replaced in order to estimate the complex backscatter coefficients and placed back after appliyng despeckling. The threshold used to select the point targets is threshold = cf · median(intensity_SAR_image). Cf is a coefficient that depends on the dataset at hand. For TerraSARX products, the point targets are identified as all intensity values above the threshold = 50 · median(intensity_SAR_image).

- Place in the training directory a bunch of 10000x10000 decorrelated complex SAR images and one in the test directory. During traning some 1000x1000 patches, extracted from the 10000x10000 test image, will be used as testing images. 

#### Usage
In _Speckle2Void-training.ipynb_ the Speckle2V object is instantiated and some parameters are required:


```
"dir_train" : directory with training data.
"dir_test"  : directory with test data.
"file_checkpoint" : checkpoint for loading a specific model. If None, the latest checkpoint is loaded.
"batch_size"  : size of the mini-batch.
"patch_size"  : size of the training patches
"model_name" : starting name of the directory where to save the checkpoints.
"lr" : learning rate.
"steps_per_epoch" : steps for each epoch 
"k_penalty_tv" : coefficient to weigh the total variation term in the loss
"norm" : normalization
"clip" : intensity value to clip the SAR images
"shift_list" : list of the possible shifts to apply to the receptive fields at the end of the network. For example [3,1].
"prob" : list of the probabilities for choosing the possible shifts. For example [0.9,0.1], 0.9 will be the probability of using shift equal to 3 and 0.1 of using shift 1.
"L_noise" : parameter L of the noise distribution gamma(L,L) used to model the speckle
```

The SAR denoiser training starts by default from the latest checkpoint found in './checkpoint/model_name' or from a specified checkpoint.

#### Checkpoint
The _s2v_checkpoint_ directory contains the model used to produce the results of the paper.

#### Testing
Download sample test images from [here](https://www.dropbox.com/s/4gfkge0pqkuylmv/decorr_complex_tsx_SLC_0.mat?dl=0) and place them in the test_examples directory.
To test the trained model on the test examples and estimated the clean versions run _Speckle2Void-prediction.ipynb_.

## Authors & Contacts
Speckle2Void is based on work by the [Image Processing and Learning](https://ipl.polito.it/) group of Politecnico di Torino: Andrea Bordone Molini (andrea.bordone AT polito.it), Diego Valsesia (diego.valsesia AT polito.it), Giulia Fracastoro (giulia.fracastoro AT polito.it), Enrico Magli (enrico.magli AT polito.it).
