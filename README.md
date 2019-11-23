# Car Detection with Neural Networks using Pytorch
our objective in this project is to detect the existence of cars in an image (binary classification)  
# Image Dataset
Vehicle image database is downloaded from https://www.gti.ssr.upm.es/data/Vehicle_database.html
The code of loading dataset is 
```python 
data_dir = '/content/drive/My Drive/Hybrid'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```
each folder contains two sub-folders /train/no-cars which is the clear image or the image without cars and /train/cars which represents the image with cars. 
![dataset images ](https://github.com/mohandesosama/car_detection/blob/master/report%20images/sample%20dataset%20classes.png)
