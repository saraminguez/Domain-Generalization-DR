# Domain Generalization for Robust Diabetic Retinopathy Classification
Domain Generalization for Robust Diabetic Retinopathy Classification is a thesis project for M.Sc. Intelligent Adaptive Systems.

In this thesis, it has been suggested to apply domain adversarial training, an unsupervised, multi-source, multi-target domain generalization model that has been proven to perform robust diabetic retinopathy classification.

## Description
Three different models have been used for this thesis: the baseline model, the joint training model, and the domain adversarial training model. The backbone for these models is ResNet-50 [1].  

In addition, the influence of the data augmentations on achieving a robust model for domain generalization has been analyzed. An ablation study of the different techniques utilized in this thesis has been performed.

A general overview of the proposed method can be seen in the following Figure: 

![Image](Images/Proposed_method.jpg)


## Dependencies 

Please install the dependencies required:

```bash
$ pip install -r requirements.txt
```

## Datasets 

The EyePACS dataset has been used to carry out the different experiments. Please download the dataset here: [EyePACS dataset](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data).

In addition, this dataset has been separated by the different cameras that were used to acquire it. The labels have been provided by Yang, Dalu, et al. [2]. It has to be mentioned that they stated differences between these labels and the ones used by them in their approach. These labels are provided in Data_preprocessing/Cameras_labels. The original dataset can be separated by cameras by using `Dataset.py`.

Moreover, bad-quality images have also been removed. These labels have been created by Fu, Huazhu, et al. [3], who studied the quality of the images, creating the Eye-Quality (EyeQ) Assessment Dataset. The labels can be downloaded here: [EyeQ Assessment Dataset](https://github.com/HzFu/EyeQ/tree/master/data). In addition, they are provided in Data_preprocessing/Quality labels. 

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
* [Messidor](https://www.adcis.net/en/third-party/messidor/) [4]
* [Messidor 2](https://www.adcis.net/en/third-party/messidor2/) [4]
* [APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) [5]


## Training models
The baseline and joint training models can be trained by running, for example, the following command:  
```
 python ResNet_model.py --images_loc '/data_a/0minguez/70-15-15/label_1' --labels_loc '/data_a/0minguez/degree_domain_labels.csv' --log_dir '/data_a/0minguez/final_experiment_joint/' --batch_size 16 --es_patience 10 --num_workers 6  --lr 1e-4  --color_transformations 'no' --mode 'train' --es_patience 10 
```
Training one model or the other, will depend on the number of datasets passed to the --images_loc argument. If only one datasets is passed, we will obtain results for the baseline model, but if more are passed then we will performed joint_training model. 

The adversarial model by running, for example, the following command:

```
python Adversarial_training.py --images_loc '/data_a/0minguez/70-15-15/label_1/' '/data_a/0minguez/70-15-15/label_2' '/data_a/0minguez/70-15-15/label_3' --labels_loc '/data_a/0minguez/degree_domain_labels.csv' --log_dir '/data_a/0minguez/adversarial_lambda/' --batch_size 16 --num_workers 6  --lr 1e-4  --color_transformations 'no' --final_checkpoint_name 'adversarial_prueba_newdisc' --mode 'train' --lambda_value 0.3
```

Data augmentations will be performed depending on the value passed to the --color_transformations argument. 'no', no data augmentation will be performed; "color,"  color augmentation will be applied, and 'augmix', AugMix transformation will be add.

## UMAP analysis
In addition to quantitative results, we have performed qualitative analysis. For this purpose, we have applied UMAP reduction to obtain 2D representation of the features extracted by the different models. 
The UMAP representation can be obtained with the following command, using `UMAP.py` or `UMAP_adversarial.py` depending on the model trained: 

```
ulimit -n 50000 && python  UMAP.py --images_loc '/data_a/0minguez/70-15-15/label_1' --labels_loc '/data_a/0minguez/degree_domain_labels.csv' --batch_size 16 --num_workers 12  --checkpoint_path '/data/0minguez/tfm-sara-minguez-retinopatia-diabetica/checkpoints/final-color.ckpt' --folder_embedding_name 'color'
```

## Author
Sara Mínguez Monedero

## References 
[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

[2] D. Yang, Y. Yang, T. Huang, B. Wu, L. Wang, and Y. Xu, “Residual-cyclegan based camera adaptation for robust diabetic retinopathy screening,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 464–474, Springer, 2020.

[3] H. Fu, B. Wang, J. Shen, S. Cui, Y. Xu, J. Liu, and L. Shao, “Evaluation of retinal image quality assessment networks in different color-spaces,” in Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part I 22, pp. 48–56, Springer, 2019.

[4] E. Decencière, X. Zhang, G. Cazuguel, B. Lay, B. Cochener, C. Trone, P. Gain, R. Ordonez, P. Massin, A. Erginay, et al., “Feedback on a publicly distributed image database: the messidor database,” Image Analysis & Stereology, vol. 33, no. 3, pp. 231–234, 2014.

[5] “Aptos dataset.” https://www.kaggle.com/competitions/aptos2019-blindness-detection/data, 2019.

