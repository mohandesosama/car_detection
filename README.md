# Car Detection with Neural Networks using Pytorch
Our objective in this project is to detect the existence of cars in an image (binary classification)  
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
For training purposes,Sotchastic Gradient Descent SGD is used in the training process with learning rate 0.006. And cross entropy loss is used as an error functions. The ephochs are 50. 
```python

optimizer=optim.SGD(model.parameters(),lr=0.006)
#optimizer=optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()
#criterion=nn.NLLLoss()
epochs = 50
print_every = 10
running_loss=0
steps=0
current_accuracy=0
for e in range(epochs):
  model.train()
  for images,labels in iter(train_loader):
    steps += 1
    images.resize_(images.size()[0],64*64*3)
    #images = images.view(images.shape[0], -1) # resize an image
    optimizer.zero_grad()
    #forward and backward passes
    output = model.forward(images)
    loss = criterion(output,labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print(loss.item())
    #print('steps ',steps)
    if steps % print_every == 0:
      model.eval()
      with torch.no_grad():
        test_loss,accuracy =validation(model,test_loader,criterion)
      print("{:.3f}".format(accuracy/len(test_loader)))
      if (accuracy / len(test_loader)) > current_accuracy:
        current_accuracy = (accuracy/len(test_loader))
        # save the model
        torch.save(model.state_dict(),'checkpoint.pth') 
      running_loss=0
      model.train()
```
To see the system performance, ROC curve is used. ROC curve plots the accuracy against recall. the following code draws the system ROC curve
```python
from sklearn import metrics
def test_class_probabilities(model, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for images, labels in test_loader:
            images.resize_(images.size()[0],64*64*3)
            output = model.forward(images)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(labels.view_as(prediction) == which_class)
            probabilities.extend(torch.exp(output[:, which_class]))
    return [i.item() for i in actuals], [i.item() for i in probabilities]

which_class = 1 # with seatbelt (positive) images
actuals, class_probabilities = test_class_probabilities(model, test_loader, which_class)

fpr, tpr, _ = metrics.roc_curve(actuals, class_probabilities)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for label=cat(%d) class' % which_class)
plt.legend(loc="lower right")
plt.show()
```
The resulting ROC curve is shown in the following image
![ROC curve](https://github.com/mohandesosama/car_detection/blob/master/report%20images/ROC%20curve.png)
Thanks  
Osama Hosameldeen  
Taibah University
