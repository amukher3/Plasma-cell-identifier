# Plasma-cell-identifier
### An effort to separate normal plasma cell images from infected Blast cells. 

### Identifying Normal blood plasma cells from Myelommic plasma cells
This is an effort to identify normal plasma cells from infected blast cell. 
The images are bone marrow images captured under optical microscope. 
The images are categorized under two categories Normal Plasma cells and Myelomma cells(Blast cells). 

***Description of the model*** 
Due to scarcity of data for one of the classes there was huge imbalance. 
Data generator came to the rescue as it was used to augment the images. 
The images were further augmented with scrapped of images from the internet 
cropped and segmented. 

The model is a simple 2-layered 2D-CONV net developed in keras. 
128 Kernels are used with dimensions of 3x3 followed by a Max-pooling layer. 
The differnet kinds of regularisation and drop-out aids the performance 
on the test-set and the prediction-set. 

***Results***
The model gets a good prediction accuracy of around 95% on the test-set.
Results were further 'sanity-checked' with a prediction set. 

***Deployment as a web UI***
The model can be currently deployed as a web-based framework locally using Flask. 
Just change the `load_model=model()` to the path where you downlaod the weights and
then use the `Cell_classifier_deployer.py` file to use it as  a local host. 

Some of the image sources are: https://sites.google.com/site/hosseinrabbanikhorasgani/datasets-1
and the Harvard data-verse. 

Commit details are in: https://bitbucket.org/amukher3/

`Bugs/queries can be reported to : abhi0787@gmail.com amukher3@rockets.utoledo.edu`



