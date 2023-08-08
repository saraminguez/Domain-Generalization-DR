# Domain Generalization for Robust Diabetic Retinopathy Classification
Domain Generalization for Robust Diabetic Retinopathy Classification is a thesis project for M.Sc. Intelligent Adaptive Systems.

In this thesis, it has been suggested to apply domain adversarial training, an unsupervised, multi-source, multi-target domain generalization model that has been proven to perform robust diabetic retinopathy classification.

## Description
Three different models have been used for this thesis: the baseline model, the joint training model, and the domain adversarial training model. 

In addition, the influence of the data augmentations on achieving a robust model for domain generalization has been analyzed.

An ablation study of the different techniques utilized in this thesis has been performed.

A general overview of the proposed method can be seen in the following Figure: 

![Image](Images/Proposed_method.jpg)


## Dependencies 

Please install the dependencies required:

```bash
$ pip install -r requirements.txt
```

## Datasets 

The EyePACS dataset has been used to carry out the different experiments. Please download the dataset here: [EyePACS dataset](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data).

In addition, this dataset has been separated by the different cameras that were used to acquire it. The labels have been provided by Yang, Dalu, et al. [1]. It has to be mentioned that they stated differences between these labels and the ones used by them in their approach. These labels are provided in Data_preprocessing/Cameras_labels. The original dataset can be separated by cameras by using `Dataset.py`.

Moreover, bad-quality images have also been removed. These labels have been created by Fu, Huazhu, et al. [2], who studied the quality of the images, creating the Eye-Quality (EyeQ) Assessment Dataset. The labels can be downloaded here: [EyeQ Assessment Dataset](https://github.com/HzFu/EyeQ/tree/master/data). In addition, they are provided in Data_preprocessing/Quality labels. 

Only a subset (28.792) of the total images were categorized by the previous dataset, therefore, the quality of the remaining images was also assessed. It was observed that certain images were still entirely dark, thus, those whose brightness fell below a predetermined threshold were additionally eliminated. This is performed by `Dark_images.ipynb`. 

In addition, images are cropped, founding the smallest bounding box in which the retina is contained and cropping the rest. The images are then resized to 512 x 512. This can be done by applying `Preprocessing_dataset.ipynb`.

Finally, the data has to have this form, where each of the cameras datasets is split into train, val, and test: 

```bash
├── your_data_dir
    ├── camera1
        ├── train
            ├── image.jpg
            ├── ...
        ├── val
            ├── image.jpg
            ├── ...
        ├── test
            ├── image.jpg
            ├── ...
    ├── camera2
    ├── camera3
    ├── ...
```
The external datasets used to evaluate the performance of the models are the following: 
* [Messidor](https://www.adcis.net/en/third-party/messidor/) [3]
* [Messidor 2](https://www.adcis.net/en/third-party/messidor2/) [3]
* [APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) [4]

## Code description

- Dataset.py: this file allows to separate the images from EyePACS dataset by their corresponding camera labels.
- Functions.py: contains different functions that will make easier perform different actions of the project. 
- Preprocessing_dataset.ipynb: Write the folder that contains the images from which we want to remove the masks and resize to 512x512, and then write the folder that we want to move the images into.


## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Author
Sara Mínguez Monedero

## References 
[1] D. Yang, Y. Yang, T. Huang, B. Wu, L. Wang, and Y. Xu, “Residual-cyclegan based camera adaptation for robust diabetic retinopathy screening,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 464–474, Springer, 2020.

[2] H. Fu, B. Wang, J. Shen, S. Cui, Y. Xu, J. Liu, and L. Shao, “Evaluation of retinal image quality assessment networks in different color-spaces,” in Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part I 22, pp. 48–56, Springer, 2019.

[3] E. Decencière, X. Zhang, G. Cazuguel, B. Lay, B. Cochener, C. Trone, P. Gain, R. Ordonez, P. Massin, A. Erginay, et al., “Feedback on a publicly distributed image database: the messidor database,” Image Analysis & Stereology, vol. 33, no. 3, pp. 231–234, 2014.

[4] “Aptos dataset.” https://www.kaggle.com/competitions/aptos2019-blindness-detection/data, 2019.

