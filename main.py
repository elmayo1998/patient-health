import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("dataset/healthcare-dataset-stroke-data.csv")

# ######################################## #
# #################### EDA ############### #
# ######################################## #

print(f"\nColonne del Dataset: \n{df.columns}\n")
print(f"\nShape: \n{df.shape}\n")
print(f"\n{df.head()}\n")
print(f"\n{df.info()}\n")
print(f"\n{df.describe()}\n")

# Rimuovo la feature "id" poiché inutile al fine dell'analisi.
df = df.drop(["id"], axis=1)

# Controllo la presenza di valori N/A.
print(f"Controllo valori N/A:\n{df.isnull().sum()}\n")

sns.heatmap(data=df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.title("Valori N/A")
plt.tight_layout()
plt.show()

# Gestisco la feature "bmi".
sns.boxplot(x='bmi', data=df)
plt.show()

# Gestisco eventuali outliers.
print(f"Valori di 'bmi' maggiori-uguali di 65: {((df.bmi >= 65).sum())}\n")

# Rimuovo i campioni con hanno un valore di "bmi" eccessivamente alto.
df.drop(df.index[df['bmi'] >= 65], inplace=True)
df.reset_index(drop=True, inplace=True)

sns.histplot(df.bmi, color='green')
plt.axvline(df['bmi'].mean(), label='mean', color='red')
plt.axvline(df['bmi'].median(), label='median', color='blue')
plt.legend()
plt.title("Distribuzione 'bmi'")
plt.tight_layout()
plt.show()

# Calcolo la media della feature "bmi" per il sesso maschile e per il femminile.
df_only_male = df["gender"] == "Male"
df_filtered_male = df[df_only_male]
print("Media 'bmi' maschile: ", df_filtered_male["bmi"].mean())
df_only_female = df["gender"] == "Female"
df_filtered_female = df[df_only_female]
print("Media 'bmi' femminile: ", df_filtered_female["bmi"].mean(), "\n")


def check_bmi(cols):
    bmi = cols[0]
    gender = cols[1]

    if pd.isnull(bmi):
        if gender == "Male":
            return df_filtered_male["bmi"].mean()
        if gender == "Female":
            return df_filtered_female["bmi"].mean()
    else:
        return bmi


df["bmi"] = df[["bmi", "gender"]].apply(check_bmi, axis=1)

print(f"Controllo i valori N/A:\n{df.isnull().sum()}\n")

sns.heatmap(data=df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.title("Valori N/A")
plt.tight_layout()
plt.show()

# Gestisco la feature "ever_married".
df["ever_married"].replace(to_replace="Yes", value=1, inplace=True)
df["ever_married"].replace(to_replace="No", value=0, inplace=True)

# Gestisco la feature "gender".
print(f"\nFrequenza e valori unici della feature 'gender':\n{df.gender.value_counts()}\n")

df.drop(df.index[df['gender'] == 'Other'], inplace=True)  # Rimuovo l'unico campione senza il sesso.
df.reset_index(drop=True, inplace=True)

df["gender"].replace(to_replace="Male", value=1, inplace=True)
df["gender"].replace(to_replace="Female", value=0, inplace=True)

print(f"\nFrequenza e valori unici della feature 'gender' aggiornata:\n{df.gender.value_counts()}\n")

# Gestisco la feature "Residence_type".
df["Residence_type"].replace(to_replace="Urban", value=1, inplace=True)
df["Residence_type"].replace(to_replace="Rural", value=0, inplace=True)

print(f"\nFrequenza e valori unici della feature 'Residence_type':\n{df.Residence_type.value_counts()}\n")


# Gestisco la feature "avg_glucose_level".
# Più che il valore esatto di glucosio nel sangue mi è utile conoscere la fascia di appartenenza.
# Preservo l'ordinamento poiché in questo caso è importante.


def check_glucose(cols):
    avg_glucose_level = cols[0]

    if avg_glucose_level < 140:
        return 0  # Normale
    if 140 <= avg_glucose_level < 200:
        return 1  # Alterato
    if avg_glucose_level >= 200:
        return 2  # Diabetico


df["avg_glucose_level"] = df[["avg_glucose_level"]].apply(check_glucose, axis=1)

sns.countplot(data=df, x="avg_glucose_level")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gestisco la feature "age".
# Anche qui, più che l'età esatta mi è utile conoscere la fascia di età.
# Preservo l'ordinamento poiché in questo caso è importante.

# Distribuisco i campioni in quattro fasce d'età.
df["age"] = pd.qcut(df["age"], q=4, labels=False)

sns.countplot(data=df, x="age")
plt.tight_layout()
plt.show()

# Gestisco la feature "smoking_status".
sns.countplot(data=df, x="smoking_status")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"\nFrequenza e valori unici della feature 'smoking_status':\n{df.smoking_status.value_counts()}\n")

# Sono presenti molti campioni con valore mancante, non posso rimuoverli.
# Calcolo in percentuale il numero di fumatori, ex fumatori e non fumatori presenti nel df.
# Escludendo dal conteggio i campioni con uno stato sconosciuto.
percentage_smokers = (df["smoking_status"] == "smokes").sum() / (len(df) - (df["smoking_status"] == "Unknown").sum())

percentage_ex_smokers = (df["smoking_status"] == "formerly smoked").sum() / (
        len(df) - (df["smoking_status"] == "Unknown").sum())

percentage_never_smokers = (df["smoking_status"] == "never smoked").sum() / (
        len(df) - (df["smoking_status"] == "Unknown").sum())

print(f"% Fumatori: {int(round(percentage_smokers * 100))}\n")  # 22%
print(f"% Ex Fumatori: {int(round(percentage_ex_smokers * 100))}\n")  # 25%
print(f"% Mai Fumatori: {int(round(percentage_never_smokers * 100))}\n")  # 53%

# Converto queste percentuali in numero di campioni da assegnare poi alla categoria "Unknown".
rows_smokers = round(((df["smoking_status"] == "Unknown").sum() * percentage_smokers))
rows_ex_smokers = round(((df["smoking_status"] == "Unknown").sum() * percentage_ex_smokers))
rows_never_smokers = round(((df["smoking_status"] == "Unknown").sum() * percentage_never_smokers))

# A questo punto so esattamente quanti campioni assegnare a ciascuna categoria.
print("Campioni da assegnare ad ogni categoria:")
print("Fumatori: ", rows_smokers)
print("Ex Fumatori: ", rows_ex_smokers)
print("Mai Fumato: ", rows_never_smokers)

# Per non assegnarli in maniera completamente casuale eseguo un ulteriore selezione.
# Salvo in un dataframe solo i campioni che hanno uno stato sconosciuto di fumatore.
df_smoking_unknown = df.loc[df["smoking_status"] == "Unknown"]

# Visualizzo le loro fasce d'età.
print("\nFrequenza e valori unici delle fasce d'età dei campioni 'Unknown': ")
print(df_smoking_unknown["age"].value_counts())

sns.countplot(data=df_smoking_unknown, x="age")
plt.title("Fasce d'età dei campioni 'Unknown'")
plt.tight_layout()
plt.show()

# Calcolo in percentuale le fasce d'età di appartenenza di questi campioni con uno stato sconosciuto.
first_quartile = (df_smoking_unknown["age"] == 0).sum() / len(df_smoking_unknown)
second_quartile = (df_smoking_unknown["age"] == 1).sum() / len(df_smoking_unknown)
third_quartile = (df_smoking_unknown["age"] == 2).sum() / len(df_smoking_unknown)
fourth_quartile = (df_smoking_unknown["age"] == 3).sum() / len(df_smoking_unknown)

print(f"% Prima Fascia: {int(round(first_quartile * 100))}\n")  # 52%
print(f"% Seconda Fascia: {int(round(second_quartile * 100))}\n")  # 18%
print(f"% Terza Fascia: {int(round(third_quartile * 100))}\n")  # 15%
print(f"% Quarta Fascia: {int(round(fourth_quartile * 100))}\n")  # 15%

# Converto queste percentuali in numero di campioni da assegnare poi rispettivamente alle tre categorie.
# Ora so esattamente quanti campioni assegnare a ciascuna fascia d'età per ogni categoria di "smoking_status".

# Distribuzione dei Fumatori per fasce d'età.
rows_smokers_0 = round((first_quartile * rows_smokers))
rows_smokers_1 = round((second_quartile * rows_smokers))
rows_smokers_2 = round((third_quartile * rows_smokers))
rows_smokers_3 = round((fourth_quartile * rows_smokers))

print(f"Distribuisco i {rows_smokers} campioni che fumano in:")
print(rows_smokers_0, rows_smokers_1, rows_smokers_2, rows_smokers_3)

# Distribuzione degli Ex Fumatori per fasce d'età.
rows_ex_smokers_0 = round((first_quartile * rows_ex_smokers))
rows_ex_smokers_1 = round((second_quartile * rows_ex_smokers))
rows_ex_smokers_2 = round((third_quartile * rows_ex_smokers))
rows_ex_smokers_3 = round((fourth_quartile * rows_ex_smokers))

print(f"\nDistribuisco i {rows_ex_smokers} campioni ex fumatori in:")
print(rows_ex_smokers_0, rows_ex_smokers_1, rows_ex_smokers_2, rows_ex_smokers_3)

# Distribuzione dei Mai Fumatori per fasce d'età.
rows_never_smokers_0 = round((first_quartile * rows_never_smokers))
rows_never_smokers_1 = round((second_quartile * rows_never_smokers))
rows_never_smokers_2 = round((third_quartile * rows_never_smokers))
rows_never_smokers_3 = round((fourth_quartile * rows_never_smokers))

print(f"\nDistribuisco gli {rows_never_smokers} campioni mai fumatori in:")
print(rows_never_smokers_0, rows_never_smokers_1, rows_never_smokers_2, rows_never_smokers_3, "\n")

# Applico al dataset i valori trovati.
# Separo in quattro dataframe differenti i campioni dividendoli per fasce d'età.
df_0 = df_smoking_unknown.loc[df_smoking_unknown["age"] == 0]

df_1 = df_smoking_unknown.loc[df_smoking_unknown["age"] == 1]

df_2 = df_smoking_unknown.loc[df_smoking_unknown["age"] == 2]

df_3 = df_smoking_unknown.loc[df_smoking_unknown["age"] == 3]

# Riempio ciascun dataframe con i valori calcolati precedentemente.
df_0.sample(frac=1, random_state=10)
a = rows_smokers_0
b = rows_ex_smokers_0
c = rows_never_smokers_0
df_0["smoking_status"].iloc[:a].replace(to_replace="Unknown", value="smokes", inplace=True)
df_0["smoking_status"].iloc[a:(a + b)].replace(to_replace="Unknown", value="formerly smoked", inplace=True)
df_0["smoking_status"].iloc[(a + b):(a + b + c + 1)].replace(to_replace="Unknown", value="never smoked", inplace=True)
# print("\n", df_0.smoking_status.value_counts())

df_1.sample(frac=1, random_state=15)
a = rows_smokers_1
b = rows_ex_smokers_1
c = rows_never_smokers_1
df_1["smoking_status"].iloc[:a].replace(to_replace="Unknown", value="smokes", inplace=True)
df_1["smoking_status"].iloc[a:(a + b)].replace(to_replace="Unknown", value="formerly smoked", inplace=True)
df_1["smoking_status"].iloc[(a + b):(a + b + c)].replace(to_replace="Unknown", value="never smoked", inplace=True)
# print("\n", df_1.smoking_status.value_counts())

df_2.sample(frac=1, random_state=20)
a = rows_smokers_2
b = rows_ex_smokers_2
c = rows_never_smokers_2
df_2["smoking_status"].iloc[:a].replace(to_replace="Unknown", value="smokes", inplace=True)
df_2["smoking_status"].iloc[a:(a + b)].replace(to_replace="Unknown", value="formerly smoked", inplace=True)
df_2["smoking_status"].iloc[(a + b):(a + b + c)].replace(to_replace="Unknown", value="never smoked", inplace=True)
# print("\n", df_2.smoking_status.value_counts())

df_3.sample(frac=1, random_state=25)
a = rows_smokers_3
b = rows_ex_smokers_3
c = rows_never_smokers_3
df_3["smoking_status"].iloc[:a].replace(to_replace="Unknown", value="smokes", inplace=True)
df_3["smoking_status"].iloc[a:(a + b)].replace(to_replace="Unknown", value="formerly smoked", inplace=True)
df_3["smoking_status"].iloc[(a + b):(a + b + c)].replace(to_replace="Unknown", value="never smoked", inplace=True)
# print("\n", df_3.smoking_status.value_counts())

# Merge to one dataset.
df_smoking_unknown = pd.concat([df_0, df_1, df_2, df_3])

df_smoking = df_smoking_unknown["smoking_status"]
# print(df_smoking)
# print(df_smoking.shape)

# Aggiorno il dataframe originale con i valori correnti calcolati.
df.update(df_smoking)

# Gestione delle feature categoriche "smoking_status" e "work_type".
print("\n", df.smoking_status.value_counts())
print("\n", df_0.work_type.value_counts())
df = pd.get_dummies(df, drop_first=True)

# Correlation Matrix.
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Percentuale feature target "stroke".
figure = (df["stroke"].value_counts() * 100.0 / len(df)) \
    .plot.pie(autopct='%.1f%%', labels=['No', 'Yes'])
figure.yaxis.set_major_formatter(tick.PercentFormatter())
plt.tight_layout()
plt.show()

print(f"{df.stroke.value_counts()}\n")

# #################################################### #
# #################### Model Selection ############### #
# #################################################### #

# Target value.
t = df["stroke"]

# Design Matrix.
X = df.drop(columns=["stroke"])

# Splitting the data into train and test. Uso il parametro stratify per mantenere la stessa distribuzione.
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, stratify=t, random_state=100)

# Normalizzo le features (i valori della feature "bmi" hanno un range diverso dalle altre).
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# # -- Logistic Regression.
# log_reg = linear_model.LogisticRegression()
#
# # Setting the range for class weights.
# weights = np.linspace(0.0, 0.99, 200)
#
# # Hyperparameter C.
# hyper_param = {
#     "C": np.logspace(-3, 3, 7),
#     "class_weight": [None, "balanced", *[{0: x, 1: 1.0 - x} for x in weights]]
# }
#
# print("\nValori di C: \n", np.logspace(-3, 3, 7), "\n")
# print("\nValori di class_weight: \n", np.linspace(0.0, 0.99, 200), "\n")
#
# log_reg_cv = GridSearchCV(log_reg, hyper_param, cv=5, scoring="f1", verbose=2)
#
# log_reg_cv.fit(X_train, t_train)
#
# print("Best C: ", log_reg_cv.best_params_)
#
# t_hat_train = log_reg_cv.predict(X_train)  # Predizione sul Train.
# print("\nAccuracy score on the train set: ", accuracy_score(t_train, t_hat_train))
# print("\nf1_score on the train set: ", f1_score(t_train, t_hat_train))
#
# # Model Assessment.
# t_hat_test = log_reg_cv.predict(X_test)
# print("\nAccuracy score on the test set: ", accuracy_score(t_test, t_hat_test))
# print("\nf1_score on the test set: ", f1_score(t_test, t_hat_test))
# print("\nClassification Report on the test set: \n", classification_report(t_test, t_hat_test))
#
# cm = confusion_matrix(t_test, t_hat_test)
# print(f"\nConfusion Matrix for Logistic Regression:\n {cm} \n")
# # print(t_test[0:50])
# # print(t_hat_test[0:50])
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title("Confusion Matrix for Logistic Regression")
# plt.show()
# #
# # -- Support Vector Machine.
# svc = SVC()
#
# # Setting the range for class weights.
# weights = np.linspace(0.0, 0.99, 200)
#
# # Hyperparameter.
# parameters = [{'C': [0.5, 1, 10, 100, 1000], 'kernel': ['linear'],
#                "class_weight": [None, "balanced", *[{0: x, 1: 1.0 - x} for x in weights]]},
#
#               {'C': [0.5, 1, 10, 50, 100, 1000], 'kernel': ['rbf'],
#                "class_weight": [None, "balanced", *[{0: x, 1: 1.0 - x} for x in weights]],
#                'gamma': ["scale", "auto", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#                "decision_function_shape": ["ovr", "ovo"]},
#
#               {'C': [0.5, 1, 10, 20, 50, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3],
#                "class_weight": [None, "balanced", *[{0: x, 1: 1.0 - x} for x in weights]],
#                'gamma': ["scale", "auto", 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
#                "decision_function_shape": ["ovr", "ovo"]},
#
#               {'C': [0.5, 1, 10, 20, 50, 100, 1000], 'kernel': ['sigmoid'],
#                "class_weight": [None, "balanced", *[{0: x, 1: 1.0 - x} for x in weights]],
#                'gamma': ["scale", "auto"]}
#               ]
#
# svc_cv = GridSearchCV(svc, parameters, cv=5, scoring="f1", verbose=2)
#
# svc_cv.fit(X_train, t_train)
#
# print("Best Parameters: \n", svc_cv.best_params_)
# print("\nBest Estimator: \n", svc_cv.best_estimator_)
#
# t_hat_train = svc_cv.predict(X_train)  # Predizione sul Train.
# print("\nAccuracy score on the train set: ", accuracy_score(t_train, t_hat_train))
# print("\nf1_score on the train set: ", f1_score(t_train, t_hat_train))
#
# # Model Assessment.
# t_hat_test = svc_cv.predict(X_test)
# print("\nAccuracy score on the test set: ", accuracy_score(t_test, t_hat_test))
# print("\nf1_score on the test set: ", f1_score(t_test, t_hat_test))
# print("\nClassification Report on the test set: \n", classification_report(t_test, t_hat_test))
#
# cm = confusion_matrix(t_test, t_hat_test)
# print(f"\nConfusion Matrix for SVC:\n {cm} \n")
# # print(t_test[0:50])
# # print(t_hat_test[0:50])
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title("Confusion Matrix for SVC")
# plt.show()

# Best Model for Logistic Regression.
log_reg = linear_model.LogisticRegression(C=10.0, class_weight={0: 0.1442713567839196, 1: 0.8557286432160804})

log_reg.fit(X_train, t_train)

print("[BEST MODEL LOGISTIC REGRESSION]\n")

t_hat_train = log_reg.predict(X_train)  # Predizione sul Train.
print("\nAccuracy score on the train set: ", accuracy_score(t_train, t_hat_train))
print("\nf1_score on the train set: ", f1_score(t_train, t_hat_train))

# Model Assessment.
t_hat_test = log_reg.predict(X_test)
print("\nAccuracy score on the test set: ", accuracy_score(t_test, t_hat_test))
print("\nf1_score on the test set: ", f1_score(t_test, t_hat_test))
print("\nClassification Report on the test set: \n", classification_report(t_test, t_hat_test))

cm = confusion_matrix(t_test, t_hat_test)
print(f"\nConfusion Matrix for Logistic Regression:\n {cm} \n")
# print(t_test[0:50])
# print(t_hat_test[0:50])
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

# Best Model for SVM Classifier.
svc = SVC(kernel="linear", C=1, class_weight={0: 0.10944723618090453, 1: 0.8905527638190954})

svc.fit(X_train, t_train)

print("[BEST MODEL SVC]\n")

t_hat_train = svc.predict(X_train)  # Predizione sul Train.
print("\nAccuracy score on the train set: ", accuracy_score(t_train, t_hat_train))
print("\nf1_score on the train set: ", f1_score(t_train, t_hat_train))

# Model Assessment.
t_hat_test = svc.predict(X_test)
print("\nAccuracy score on the test set: ", accuracy_score(t_test, t_hat_test))
print("\nf1_score on the test set: ", f1_score(t_test, t_hat_test))
print("\nClassification Report on the test set: \n", classification_report(t_test, t_hat_test))

cm = confusion_matrix(t_test, t_hat_test)
print(f"\nConfusion Matrix for SVC:\n {cm} \n")
# print(t_test[0:50])
# print(t_hat_test[0:50])
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix for SVC")
plt.show()

# Provo a bilanciare il dataset con una tecnica di campionamento sintetico.
# Per vedere come si comporta Logistic Regression.

# Bilancio solo il Training Set.
smt = SMOTE(sampling_strategy="not majority", random_state=300)
X_train, t_train = smt.fit_resample(X_train, t_train)

sns.countplot(x=t_train, data=df)
plt.title("t train bilanciata")
plt.show()

# -- Logistic Regression.
log_reg = linear_model.LogisticRegression()

# Hyperparameter C.
hyper_param = {
    "C": np.logspace(-3, 3, 7),
}

print("\nValori di C: \n", np.logspace(-3, 3, 7), "\n")

log_reg_cv = GridSearchCV(log_reg, hyper_param, cv=5, scoring="f1", verbose=2)

log_reg_cv.fit(X_train, t_train)

print("Best C: ", log_reg_cv.best_params_)

t_hat_train = log_reg_cv.predict(X_train)  # Predizione sul Train.
print("\nAccuracy score on the train set: ", accuracy_score(t_train, t_hat_train))
print("\nf1_score on the train set: ", f1_score(t_train, t_hat_train))

# Model Assessment.
t_hat_test = log_reg_cv.predict(X_test)
print("\nAccuracy score on the test set: ", accuracy_score(t_test, t_hat_test))
print("\nf1_score on the test set: ", f1_score(t_test, t_hat_test))
print("\nClassification Report on the test set: \n", classification_report(t_test, t_hat_test))

cm = confusion_matrix(t_test, t_hat_test)
print(f"\nConfusion Matrix for Logistic Regression:\n {cm} \n")
# print(t_test[0:50])
# print(t_hat_test[0:50])
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
