import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing


def is_child(age):
	if age < 16:
		return 1
	else:
		return 0



if __name__ == '__main__':

    data_train = pd.read_csv("/Users/wangruidong/Newcastle University/Machine Learning/train.csv")
    data_test = pd.read_csv("/Users/wangruidong/Newcastle University/Machine Learning/test.csv")
    df = pd.concat([data_train, data_test])

    fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(15, 5))
    axis1.set_title('Original Age values')
    axis2.set_title('New Age values')
    average_age = df["Age"].mean()
    std_age = df["Age"].std()
    count_nan_age = df["Age"].isnull().sum()
    rand = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)
    df['Age'].plot(kind='hist', bins=70, ax=axis1)
    df['Age'][df.Age.isnull()] = rand
    df['Age'].plot(kind='hist', bins=70, ax=axis2)

    df['Child'] = df['Age'].apply(is_child)
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
    seaborn.countplot(x='Child', data=df, ax=axis1)
    child_survive = df[["Child", "Survived"]].groupby(['Child'], as_index=False).mean()
    seaborn.barplot(x='Child', y='Survived', data=child_survive, ax=axis2)

    df = df.drop(['Cabin'], axis=1)

    df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
    input_df_tmp = df[:data_train.shape[0]]
    (s, c, q) = df['Embarked'].value_counts()
    embark_percentage = pd.DataFrame({
        'Embarked': np.array(['S', 'C', 'Q']),
        'percentage': np.array([float(i) / df['Embarked'].count() for i in (s, c, q)])})
    fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 5))
    seaborn.barplot(x='Embarked', y='percentage', data=embark_percentage, ax=axis1)
    seaborn.countplot(x='Survived', hue="Embarked", data=input_df_tmp, order=[1, 0], ax=axis2)
    embark_perc = input_df_tmp[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
    seaborn.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)

    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    fare_not_survived = df["Fare"][df["Survived"] == 0]
    fare_survived = df["Fare"][df["Survived"] == 1]
    avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
    df['Fare'].plot(kind='hist', figsize=(15, 3), bins=100, xlim=(0, 50))
    avgerage_fare.index.names = ["Survived"]
    avgerage_fare.plot(kind='bar', legend=False)

    input_df_tmp = df[:data_train.shape[0]]
    seaborn.factorplot('Pclass', 'Survived', order=[1, 2, 3], data=input_df_tmp, size=6)

    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 5))
    seaborn.countplot(x='Sex', data=df, ax=axis1)
    women_survive = df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
    seaborn.barplot(x='Sex', y='Survived', data=women_survive, ax=axis2)

    df['Sex'][df['Sex'] == 'male'] = 1
    df['Sex'][df['Sex'] == 'female'] = 0
    df['Sex'] = df['Sex'].astype(int)

    df['WithFamily'] = df["Parch"] + df["SibSp"]
    df['WithFamily'].loc[df['WithFamily'] > 1] = 1
    df['WithFamily'].loc[df['WithFamily'] == 0] = 0

    input_df_tmp = df[:data_train.shape[0]]
    fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
    seaborn.countplot(x='WithFamily', data=df, order=[1, 0], ax=axis1)
    family_perc = input_df_tmp[["WithFamily", "Survived"]].groupby(['WithFamily'], as_index=False).mean()
    seaborn.barplot(x='WithFamily', y='Survived', data=family_perc, order=[1, 0], ax=axis2)
    axis1.set_xticklabels(["With Family", "Alone"], rotation=0)
    # plt.show()
    # df.info()
    # scaler = preprocessing.StandardScaler()
    # df['Fare_scaled'] = scaler.fit_transform(df['Fare'])
    # df = df.drop(['Fare'], axis=1)
    # df['Age_scaled'] = scaler.fit_transform(df['Age'])
    # df = df.drop(['Age'], axis=1)


    # #random forest
    # X = df[:data_train.shape[0]].values[:, 1::]
    # y = df[:data_train.shape[0]].values[:, 0]
    #
    # X_test = df[data_train.shape[0]:].values[:, 1::]
    # random_forest = RandomForestClassifier()
    # random_forest.fit(X, y)
    #
    #
    # Y_pred = random_forest.predict(X_test)
    # print random_forest.score(X, y)
    # submission = pd.DataFrame({
    #     "PassengerId": X_test["PassengerId"],
    #     "Survived": Y_pred.astype(int)
    # })
    # print type(submission) #['Survived'].sum
    # # print data_test['Survived'].sum()

    print df.info()

    X = df[:data_train.shape[0]].values[:, 1::]
    y = df[:data_train.shape[0]].values[:, 0]

    X_test = df[data_train.shape[0]:].values[:, 1::]
    GBDT = GradientBoostingClassifier(n_estimators=1000)
    GBDT.fit(X, y)

    Y_pred = GBDT.predict(X_test)
    print GBDT.score(X, y)
    # submission = pd.DataFrame({
    #     "PassengerId": X_origin["PassengerId"],
    #     "Survived": Y_pred.astype(int)
    # })
    # submission.to_csv('result.csv', index=False)





