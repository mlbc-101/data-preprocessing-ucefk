""" Do not delete this section, Please Commit your changes after implementing the necessary code.

- The data file called Social_Network_Ads.csv.
- Your Job is to preprocess this data because we gonna use it later one in the course.

The Features of this dataset are:
	- UserID: Which represent id of user in the database.
	- Gender: Can be male or female.
	- EstimatedSalary: The salary of the user.
	- Purchased: An integer number {1 if the user purshased something, 0 otherwise}
	
	The target variable for this data is the purshased status.

Happy coding."""

# Step 0: import the necessary libraries: pandas, matplotlib.pyplot, and numpy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: load your dataset using pandas

data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:,1:4].values
y = data.iloc[:,-1].values

# Step 2: Handle Missing data if they exist.

# no misssing values

# Step 3: Encode the categorical variables.

from sklearn.preprocessing import LabelEncoder

label_sexe = LabelEncoder()

X[:,0] = label_sexe.fit_transform(X[:,0])

# Step 4: Do Feature Scaling if necessary.

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)

# Final Step: Train/Test Splitting.

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , y,test_size = 0.25 , random_state = 2424)