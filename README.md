# Linear Regression Health Costs Calculator
## **Step-by-Step Explanation and Code**

---

### **1. Importing Libraries**

**What?**  
- We import the required libraries for data handling, visualization, and deep learning.  

**How?**  
```python
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
```

**Why?**  
- Libraries like `pandas` handle datasets, `matplotlib` is used for visualization, and TensorFlow/Keras provides tools for building and training machine learning models.
- `tensorflow_docs` is used for plotting and documentation, enhancing visualization of results.

---

### **2. Importing the Dataset**

**What?**  
- Load the dataset from a provided URL.  

**How?**  
```python
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()
```

**Why?**  
- The `wget` command fetches the dataset, and `pd.read_csv` reads the CSV file into a Pandas DataFrame.
- `.tail()` displays the last few rows of the dataset, giving a quick view of its structure.

---

### **3. Exploring and Preprocessing the Data**

#### a. **Inspecting the Dataset**  
**What?**  
- View the dataset's structure, check for missing values, and understand the features.

**How?**  
```python
print(dataset.head())
print(dataset.info())
print(dataset.describe())
```

**Why?**  
- Helps us understand the types of data (e.g., categorical or numerical) and identify potential preprocessing steps (e.g., encoding or scaling).

#### b. **Encoding Categorical Variables**  
**What?**  
- Convert categorical features like `sex`, `smoker`, and `region` into numeric format.  

**How?**  
```python
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], drop_first=True)
```

**Why?**  
- Machine learning models cannot work directly with categorical data, so encoding is necessary.

#### c. **Splitting Features and Labels**  
**What?**  
- Separate the independent variables (`features`) and the dependent variable (`expenses`).  

**How?**  
```python
labels = dataset.pop('expenses')
features = dataset
```

**Why?**  
- Separating labels is crucial for supervised learning, where the model learns to predict labels based on features.

---

### **4. Splitting the Dataset into Training and Testing Sets**

**What?**  
- Divide the data into training (80%) and testing (20%) sets.  

**How?**  
```python
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42)
```

**Why?**  
- Training data is used to build the model, while testing data evaluates its performance on unseen data.

---

### **5. Scaling the Data**

**What?**  
- Standardize the features to have a mean of 0 and a standard deviation of 1.  

**How?**  
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
```

**Why?**  
- Scaling ensures all features contribute equally to the model, avoiding bias due to differing magnitudes.

---

### **6. Building and Training the Model**

**What?**  
- Create a linear regression model using TensorFlow/Keras.  

**How?**  
```python
model = keras.Sequential([
    layers.Dense(1, input_shape=(train_features.shape[1],))
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(train_features, train_labels, epochs=50, validation_split=0.2, verbose=1)
```

**Why?**  
- The single-layer dense model performs linear regression.
- `adam` optimizer ensures efficient learning, `mse` (mean squared error) minimizes prediction errors, and `mae` (mean absolute error) evaluates performance.

---

### **7. Evaluating the Model**

**What?**  
- Check the model’s performance on the testing data using `model.evaluate`.  

**How?**  
```python
loss, mae, mse = model.evaluate(test_features, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))
```

**Why?**  
- Evaluating ensures the model generalizes well to unseen data, and the MAE must be under 3500 to pass.

---

### **8. Visualizing Predictions**

**What?**  
- Plot true vs. predicted healthcare expenses.  

**How?**  
```python
test_predictions = model.predict(test_features).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()
```

**Why?**  
- A scatterplot helps visually assess the model’s accuracy. Points should cluster near the diagonal line if predictions are close to true values.

---

### **Complete Markdown File**

Here’s the Markdown file content to document everything.

```markdown
# Linear Regression Health Costs Calculator

## **Objective**
Predict healthcare costs using a regression model and achieve a Mean Absolute Error (MAE) under 3500.

---

## **Steps**

### **1. Importing Libraries**
```python
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
```

### **2. Loading the Dataset**
```python
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()
```

### **3. Data Exploration and Preprocessing**
#### a. Inspecting Data
```python
print(dataset.head())
print(dataset.info())
print(dataset.describe())
```

#### b. Encoding Categorical Variables
```python
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], drop_first=True)
```

#### c. Splitting Features and Labels
```python
labels = dataset.pop('expenses')
features = dataset
```

### **4. Splitting Dataset**
```python
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42)
```

### **5. Scaling Data**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
```

### **6. Building and Training the Model**
```python
model = keras.Sequential([
    layers.Dense(1, input_shape=(train_features.shape[1],))
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(train_features, train_labels, epochs=50, validation_split=0.2, verbose=1)
```

### **7. Evaluating the Model**
```python
loss, mae, mse = model.evaluate(test_features, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))
```

### **8. Visualizing Predictions**
```python
test_predictions = model.predict(test_features).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()
```

---

## **Conclusion**
The model should achieve an MAE below 3500 and predict healthcare costs effectively. The scatterplot provides insights into prediction accuracy.
```
