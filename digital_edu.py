import pandas as pd
df = pd.read_csv('train.csv')
df.info()

df.drop(['id', 'has_photo', 'has_mobile', 'followers_count','life_main', 'people_main', 'graduation','last_seen', 'career_start', 'career_end', 'city'], axis=1, inplace=True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df['education_form'].fillna('Full-time', inplace = True)
df.drop('education_form', axis = 1, inplace = True)

def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == 'Student (Specialist)' or edu_status == 'Student (Bachelor`s)' or edu_status == 'Student (Master`s)':
        return 1 
    elif edu_status == 'Alumnus (Master`s)' or edu_status == 'Alumnus (Specialist)' or edu_status == 'Alumnus (Bachelor`s)':
        return 2
    elif edu_status == 'PhD' or edu_status == 'Candidate of Sciences':
        return 3


df['education_status'] = df['education_status'].apply(edu_status_apply)
df.info()

def split_langs(langs):
    return langs.split(';')
df['langs'] = df['langs'].apply(split_langs)
df['langs'] = df['langs'].apply(len)
print(df['langs'].value_counts())

df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandartScaler
from sklearn.neighbors import KNeighborsClassfier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop('result', axis = 1)
y = df[result]
x_Train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25) 

sc = StandartScaler()
x_Train = sc.fit_transform(x_Train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassfier(n_neighbors = 5)
classifier.fit(x_Train, y_train)

y_pred = classifier.predict(x_test)
print('% Правильности предскаания:', round(accuracy_score(y_test, y_pred) * 100, 2))
# print('Confusion matrix:')
# print(confusion_matrix(y_test, y_pred))










