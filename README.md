# Developing a Neural Network Regression Model
## Reg no: 212224230213
## Name: Rabin R

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

The experiment aims to understand the impact of different hyperparameters such as learning rate, number of hidden layers, and activation functions on model performance. It also focuses on improving the model’s generalization ability by applying techniques like normalization, regularization, and validation. Finally, the experiment helps in gaining practical knowledge of implementing deep learning models for real-world regression problems.

## Neural Network Model

<img width="985" height="597" alt="Screenshot 2026-04-24 112940" src="https://github.com/user-attachments/assets/80b2114e-aec4-478f-a82e-b2691340fae0" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing.

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot.

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: BAUDHIGAN D

### Register Number: 212223230028

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
    def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
    lig=NeuralNet()
    criterion=nn.MSELoss()
    optimizer=optim.RMSprop(lig. parameters(),lr=0.001)

    def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
        #Include your code here
        for epoch in range(epochs):
           optimizer.zero_grad()
           loss=criterion(ai_brain(X_train),y_train)
           loss.backward()
           optimizer.step()
           ai_brain.history['loss'].append(loss.item())
           if epoch % 200 == 0:
              print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

    train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)
```

### Dataset Information
<img width="319" height="272" alt="image" src="https://github.com/user-attachments/assets/522644f8-002c-4131-833d-755b00510635" />




### OUTPUT

### Training Loss Vs Iteration Plot
<img width="758" height="593" alt="image" src="https://github.com/user-attachments/assets/9227d352-0358-412b-9e63-23bdb04ead76" />



### New Sample Data Prediction
<img width="312" height="37" alt="image" src="https://github.com/user-attachments/assets/1394d967-02c3-4e7e-affa-e165c504cbaa" />





## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
