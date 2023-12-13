---
{"dg-publish":true,"permalink":"/coding/python/mnist-dataset/","tags":["python","pandas","matplotlib","pytorch"]}
---

```python
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from torch import optim
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
```

```python
df = pd.read_csv('train.csv')
df
```

|label|pixel0|pixel1|pixel2|pixel3|pixel4|pixel5|pixel6|pixel7|pixel8|...|pixel774|pixel775|pixel776|pixel777|pixel778|pixel779|pixel780|pixel781|pixel782|pixel783|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|1|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|1|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|2|1|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|3|4|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|4|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
|41995|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|41996|1|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|41997|7|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|41998|6|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|41999|9|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|

```
42000 rows × 785 columns
```

```python
labels_tensor = torch.tensor(df['label'].values, dtype=torch.long)

pixels_tensor = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
scaler = StandardScaler()
pixels_tensor = torch.tensor(scaler.fit_transform(pixels_tensor), dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(pixels_tensor, labels_tensor, test_size=0.2, random_state=42)

X_train = X_train.view(-1, 1, 28, 28)
X_test = X_test.view(-1, 1, 28, 28)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

```python
class CNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.AvgPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(432, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

model = CNN()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
```

```
SimpleCNN(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu): ReLU()
  (maxpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (conv2): Conv2d(6, 12, kernel_size=(3, 3), stride=(1, 1))
  (relu2): ReLU()
  (maxpool2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=432, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

```python
def train(epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
```

```python
train(epochs=30)
```

```
Epoch 1/30, Loss: 0.0291155388457368
Epoch 2/30, Loss: 0.0265785177116881
Epoch 3/30, Loss: 0.021704431599715635
Epoch 4/30, Loss: 0.018003975514411217
Epoch 5/30, Loss: 0.018216826700622083
Epoch 6/30, Loss: 0.016057611304983895
Epoch 7/30, Loss: 0.016210283352106454
Epoch 8/30, Loss: 0.012766154168084973
Epoch 9/30, Loss: 0.012294081108318226
Epoch 10/30, Loss: 0.01216488897795595
Epoch 11/30, Loss: 0.012132637557058063
Epoch 12/30, Loss: 0.009550272665804924
Epoch 13/30, Loss: 0.00713989697756049
Epoch 14/30, Loss: 0.006665604077069951
Epoch 15/30, Loss: 0.009995190314125683
Epoch 16/30, Loss: 0.011534767256945045
Epoch 17/30, Loss: 0.004749612184794473
Epoch 18/30, Loss: 0.006365367541648517
Epoch 19/30, Loss: 0.006927345956475099
Epoch 20/30, Loss: 0.006737604219893001
Epoch 21/30, Loss: 0.005423978301561787
Epoch 22/30, Loss: 0.0053279530611949405
Epoch 23/30, Loss: 0.004246399002526492
Epoch 24/30, Loss: 0.008340141954708672
Epoch 25/30, Loss: 0.0061150758662638725
...
Epoch 27/30, Loss: 0.002711675244282016
Epoch 28/30, Loss: 0.005945048703716956
Epoch 29/30, Loss: 0.006267678272375374
Epoch 30/30, Loss: 0.0055955590596930786
```

```python
def validate():
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predictions == labels).sum().item()

    accuracy = total_correct / total_samples
    return accuracy
```

```python
accuracy = validate()
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
```

```
Validation Accuracy: 98.69%
```
