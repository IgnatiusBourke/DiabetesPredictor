import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Import the Prima Indian data and print any missing values
data = pd.read_csv("prima-indians-diabetes-annotated.csv")
print(data.apply(lambda x: sum(x.isnull()),axis=0))

# Define the dependent and independent variables
X = data[["number_times_pregnant",'plasma_glucose_concentration','diastolic_bp','triceps_skin_fold','2hr_serum_insulin',
          'BMI','diabetes_pedigree_function','age']]
Y = data["diabetes_mellitus"]


#Define the training and testing data for the X and Y variables
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20, random_state=101)

#Define reg with the logistic regression model
reg = LogisticRegression()

# Define the parameters of the inputs in a grid
param_grid= [
    {'penalty':['l1','l2','elasticnet','none'],
    'C':np.logspace(-4,4,20),
    'solver':['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter':[100,1000,2500,5000]
    }
]

#Use grid sratch to identify the best input parameters for the model
grid_search = GridSearchCV(reg, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
best_grid = grid_search.fit(X,Y)

# Print the best model and create a new logistic regression model with the new parameters
best_grid.best_estimator_
reg_model = LogisticRegression(C=0.0001, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='none',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

#Fit the model to the training data
reg_model.fit(X_train,y_train)

#Parameters accuracy testing, best_grid produces a model with better accuracy
print(f'Train Accuracy: {best_grid.score(X_train,y_train):.3f}')
print(f'Test Accuracy: {best_grid.score(X_test,y_test):.3f}')

#Prediction of the test results
y_pred = best_grid.predict(X_test)

#Save model using Pickle
import pickle

pickle_out = open('V2_logreg_model.pk1',"wb")
pickle.dump(best_grid,pickle_out)
pickle_out.close()


########### VISUALISATION ################
#Print the decision matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

#visualise confusion matrix
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')