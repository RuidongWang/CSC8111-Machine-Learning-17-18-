import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,make_scorer

#this code reference by https://www.kaggle.com/altafang/titanic-data-analysis/notebook
def prepare_features(data, training=True):
    # Encode categorical features
    if training:
        prepare_features.sex_enc = LabelEncoder()  # Attribute of this function
        prepare_features.sex_enc.fit(data['Sex'])
    data['Sex'] = prepare_features.sex_enc.transform(data['Sex'])

    # get_dummies does one hot encoding and automatically handles nan's by making them all 0
    data = data.join(pd.get_dummies(data['Embarked']))
    data['Age'] = data['Age'].fillna(value=data['Age'].mean())
    data['Fare'] = data['Fare'].fillna(value=data['Fare'].mean())

    # Add name length as a feature to replace name (idea from a public kernel on kaggle)
    data['Name Length'] = data['Name'].apply(len)

    # Choose features to consider. Last few are from embarked
    features = ['Age', 'Sex', 'Fare', 'Pclass', 'SibSp', 'S', 'C', 'Q']

    X = np.array(data[features])  # Features

    # Scale features
    if training:
        prepare_features.scaler = StandardScaler()
        prepare_features.scaler.fit(X)
    X = prepare_features.scaler.transform(X)

    return X




# Read in data
train_data = pd.read_csv("/Users/wangruidong/Newcastle University/Machine Learning/train.csv")

X = prepare_features(train_data, training=True)
y = np.array(train_data['Survived'])  # Outcomes

# Split to get a cross-validation set to better estimate performance
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2, random_state=42)


print "===============this for svm"
clf = SVC()

# Train the model
clf.fit(X_train, y_train)

predicted_y = clf.predict(X_cv)
print("accuracy of test: ", accuracy_score(y_cv, predicted_y))
print type(y_cv)
predicted_y = clf.predict(X_train)
print("accuracy of training: ", accuracy_score(y_train, predicted_y))


# Better method for cross-validation: multiple splits
# https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html
scores = cross_val_score(clf, X, y, cv=4)
print(scores.mean())

param_grid = [{'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01]}]

svc = SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(X, y)
# Check what the best parameters were
print(clf.best_params_)
model = clf.best_estimator_
model = model.fit(X, y)
scores = cross_val_score(model, X, y, cv=4)
print "=================this for SVM"
print "the accuracy of the cross validation score : ", scores.mean()

test_set = pd.read_csv("/Users/wangruidong/Newcastle University/Machine Learning/test.csv")

X_test = prepare_features(test_set, training=False)

y_test = model.predict(X_test)
predictions = pd.DataFrame(data={'Survived': y_test}, index=test_set['PassengerId'])

print "the accuracy of the test set : ", accuracy_score(test_set['Survived'], y_test)
print "the mean f1 score of the test set : ", metrics.f1_score(test_set['Survived'], y_test, average='weighted')

print "=================this for RF"
parameters = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
acc_scorer = make_scorer(accuracy_score)
clf = RandomForestClassifier()
rf_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
rf_obj = rf_obj.fit(X_train, y_train)


clf = rf_obj.best_estimator_
clf = clf.fit(X_train, y_train)
test_predictions = clf.predict(X_cv)
print "the accuracy of the cross validation score : ", accuracy_score(y_cv, test_predictions)

pridctions = clf.predict(X_test)
print "the accuracy of the test set : ", accuracy_score(test_set['Survived'], pridctions)
print "the mean f1 score of the test set : ", metrics.f1_score(test_set['Survived'], pridctions, average='weighted')






