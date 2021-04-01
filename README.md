# Project MAG3 AMSE : "Hands-on-2021"

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)      [![forthebadge](https://forthebadge.com/images/badges/made-with-crayons.svg)](https://forthebadge.com)

### Description of the project : 
Build two models (**Neural Netowrk and Support Vector Machine**) and **an interface** to classify traffic signs (GTSRB dataset)

## To start the project properly
* clone this repository
* download images with 'scripts/download_images.sh'
* download constants.py files with the label of each classes : the name of the panels
* check requirements files : 
    - **requirements_NN.txt** is useful for the reading_images notebook
    - **requirements_SVM.txt** is useful for the SVM notebook


## To reproduce models 
* for the Neural network :
    - the whole code is on the folder notebook and it's **reading_images.ipynb**
    - you can also just load the model : it is on the models folder and it's **traffic_signs_2021-00-29_20-00-25.h5** 
* for the SVM : 
    - the whole code is on the folder notebook and it's **SVM.ipynb**
    - you can also just load the model : it is on the models folder and it's **SVM-traffic.jonlib** (be careful the template is very heavy and may take some time to download) 


## To reproduce the application 
* You need to download the application folder with :
    - a parameters file app.yaml read at the start of the application containing the path to the file, the resize of the images 
    - a python file with all the code 


## Manipulation to be done directly on the computer environment : 
If you are using a Google collab for the neural network model and you want to reload the model on a Jupyter lab, you have to proceed in the same way : 
1. From an anaconda prompt, create a new environnement 
`` conda create --name traffic python=3.8``
`` conda install pandas tensorflow scikit-learn seaborn pillow``
2. Activate this environnement 
``conda activate traffic``
3. Launch python 
``from tensorflow import keras``
``classifier = keras.models.load_model('traffic_signs_2021-03-19_13-51-00.h5') classifier.summary()`` 
4. Install the module that allows Jupyter to talk with the Python environment
``conda install ipykernel``
``python -m ipykernel install --user --name=traffic``

For the application, we have to install Dash : 
1. Install Dash in the traffic environment, as well as the Bootstrap components for dash :
``conda install dash``
`` conda install -c conda-forge dash-bootstrap-components ``

## Auteurs
- Projet réalisé par Annabelle Filip et Florine Pritzy, classe de MAG3, Aix Marseille School of Economics
- Projet réalisé dans le cadre du cours de Big Data, de Monsieur Mignot. 

## References 

* "Dataset introduction : https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
- Dataset created by Christian Igel
* "Images : https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370s/published-archive.html
