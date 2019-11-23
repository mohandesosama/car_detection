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
The images are standarized and normalized with the following code 
```python
train_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), 
                                                           (0.5, 0.5, 0.5))
                                     ])

cost_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], 
                                                           [0.5, 0.5, 0.5])
                                     ])

test_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], 
                                                           [0.5, 0.5, 0.5])
                                     ])
```
Normalization is needed to make faster and stable training process.  
The following code creates a deep neural network with 64*64*3 by 512 by 256 by 138 by 2. The last 2 is the output layer that represents with/without cars output. 
```python 

input_size = 64*64*3 # 64 pixels and 3 channels RGB
hidden_sizes=[512,256,128]
output_size=2
model=nn.Sequential(nn.Linear(input_size,hidden_sizes[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[0],hidden_sizes[1]),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_sizes[1],output_size),
                    nn.Softmax(dim=1))

print('model created.')
model
```
The validation function are used in the training process and define the best point for the training process to stop
```python

def validation(model1,testloader1,criterion1):
  test_loss=0
  accuracy = 0
  steps=0
  for images1,labels1 in testloader1:
    steps += 1
    #print('test step ',steps)
    images1.resize_(images1.size()[0],64*64*3)
    output=model1.forward(images1)
    test_loss += criterion1(output,labels1).item()
    # ps=probabilites. 
    ps=torch.exp(output)
    equality = (labels1.data==ps.max(dim=1)[1])
    accuracy += equality.type(torch.FloatTensor).mean()
  return test_loss,accuracy
  ```
