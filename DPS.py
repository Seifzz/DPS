import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVR


# Reading and acquiring some info
df = pd.read_csv("220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv")
print(df.info)
print('Missing_values:\n', df.isna().sum())

# Removing unnecessary rows
df = df[df['MONAT'] != 'Summe']
df = df[(df.JAHR != 2022) & (df['JAHR'] != 2021)]

def Vis():
    # Visualizing the yearly number of accidents per category
    a = df.groupby(['MONATSZAHL', 'JAHR'], as_index=False).sum()

    Alk = a.loc[(a['MONATSZAHL'] == 'Alkoholunfälle')]
    Flu = a.loc[(a['MONATSZAHL'] == 'Fluchtunfälle')]
    Ver = a.loc[(a['MONATSZAHL'] == 'Verkehrsunfälle')]

    x_indexes = np.arange(len(Alk))
    width = 0.25

    plt.bar(x_indexes - width, Alk.iloc[:,2], width=width, color="#444444", label = 'Alkoholunfälle')
    plt.bar(x_indexes, Flu.iloc[:,2], width=width, color="#008fd5", label = 'Fluchtunfälle')
    plt.bar(x_indexes + width, Ver.iloc[:,2],width=width, color="#e5ae38", label = 'Verkehrsunfälle')

    plt.legend()
    plt.xticks(ticks=x_indexes, labels=Alk.iloc[:,1], rotation= 45)
    plt.title('Yearly number of accidents per category')
    plt.tight_layout()
    plt.savefig("Yearly number of accidents per category")
    plt.show()


    # X_Alk = df.loc[(df['MONATSZAHL'] == 'Alkoholunfälle') & (df['MONAT'] == 'Summe') , 'JAHR']
    # y_Alk = df.loc[(df['MONATSZAHL'] == 'Alkoholunfälle') & (df['MONAT'] == 'Summe'), 'WERT']

    # y_Flu = df.loc[df['MONATSZAHL'] == 'Fluchtunfälle', 'WERT']

    # y_Ver = df.loc[df['MONATSZAHL'] == 'Verkehrsunfälle', 'WERT']

# Vis()

##################################################################

def Reg():
    # Label Enconding by categorizing the string values of two columns
    df["MONATSZAHL"] = df["MONATSZAHL"].astype('category')
    df["MONATSZAHL_cat"] = df["MONATSZAHL"].cat.codes
    print('\nLabel codes for MONATSZAHL:\n', dict(enumerate(df["MONATSZAHL"].cat.categories)))

    df["AUSPRAEGUNG"] = df["AUSPRAEGUNG"].astype('category')
    df["AUSPRAEGUNG_cat"] = df["AUSPRAEGUNG"].cat.codes
    print('\nLabel codes for AUSPRAEGUNG:\n', dict(enumerate(df["AUSPRAEGUNG"].cat.categories)))

    # Fitting the model and forecasting the target
    X = df[['MONAT', 'MONATSZAHL_cat', 'AUSPRAEGUNG_cat']]
    y = df.iloc[:, 4]

    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, y)

    y_pred = regressor.predict([['202101', '0', '1']])
    print('\nprediction: ', round(y_pred[0]))
Reg()