import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
from forex_python.converter import CurrencyRates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
'''Nettoyage. Puis se pose la question de la préparation des données proprement dite. Ces
données nécessitent-elles un nettoyage ? Faut-il écarter certaines instances qui ne sont
pas liées au problème ? Y a-t-il des valeurs manquantes ? Des valeurs aberrantes ? Des
attributs redondants ? Des attributs superflus ? Les valeurs numériques correspondent-elles
vraiment à des attributs de nature numérique, ordinale, ou catégorielle ? Comment traiter
ces différents problèmes ?'''



'''
Recodage. En fonction des algorithmes de fouille que vous allez appliquer, il peut être nécessaire de recoder certains champs : discrétisation d’attributs réels, catégorisation d’attributs
numériques, normalisation d’attributs numériques, numérisation d’attributs catégoriels... Certains outils ne peuvent pas du tout être appliqués sur des données dont le codage n’est
pas approprié. D’autres fonctionneront mieux pour certains codages. Il est recommandé de
tester l’effet du codage sur les différents outils considérés

'''


# le nom du fichier des données à utiliser pour le projet donné
path = "projects.csv"
scaler = MinMaxScaler()


data = pd.read_csv(path, encoding="ISO-8859-1")


class DataPreparation:
    def __init__(self, data):
        self.data = data
        self.instance_to_convert = ["sex", "category", "subcategory","state"]
        self.c = CurrencyRates()
    def currency_calculator(self, row_currency, row_goal, converter_date):
        
        dt = datetime.datetime.strptime(converter_date,'%Y-%m-%d %H:%M:%S' )
        if row_currency== "USD":
            exchange =self.c.get_rate('USD', 'EUR', dt)
            euro = exchange * row_goal
            return euro
        elif row_currency == "CAD":
            exchange =self.c.get_rate('CAD', 'EUR', dt)
            euro = exchange * row_goal
            return euro
        elif row_currency == "GBP":
            exchange =self.c.get_rate('GBP', 'EUR',dt)
            euro = exchange * row_goal
            return euro

        elif row_currency == "AUD":
            exchange =self.c.get_rate('AUD', 'EUR',dt)
            euro = exchange * row_goal
            return euro
        elif row_currency == "DKK":
            exchange =self.c.get_rate('DKK', 'EUR',dt)
            euro = exchange * row_goal
            return euro
        elif row_currency == "SEK":
            exchange =self.c.get_rate('SEK', 'EUR',dt)
            euro = exchange * row_goal
            return euro

        elif row_currency == "NOK":
            exchange =self.c.get_rate('NOK', 'EUR',dt)
            euro = exchange * row_goal
            return euro
        elif row_currency =="NZD":
            exchange =self.c.get_rate('NZD', 'EUR',dt)
            euro = exchange * row_goal
            return euro
        elif row_currency == "CHF":
            exchange =self.c.get_rate('CHF', 'EUR',dt)
            euro = exchange * row_goal
            return euro

        else:
            euro = row_goal
            return euro 
        
            

    def nettoyage(self):
        # drop the nan values 
        print("Dropping the nan values...")
        data = self.data.dropna()
        data = data[:100]
        print("Dropping completed")
        # Dropping the useless attributes
        print("Convert columns into numerical values")
        for idx, instance in enumerate(self.instance_to_convert):
            unique_elements = data[instance].unique()
            unique_label = dict()
            for id, element in enumerate(unique_elements):
                unique_label[element] =id
            data[instance] = data[instance].apply(lambda x: unique_label[x])
        print("End of conversion")
        print("=========================Start currency conversion ============================")

        data['goals_euro'] = data[["currency","goal","start_date"]].apply(lambda x:self.currency_calculator(*x),axis=1)
        data['pledged_euro'] = data[["currency","pledged","end_date"]].apply(lambda x:self.currency_calculator(*x), axis=1)
        print("="*10, "End", "="*10)

        data.drop(["goal", "pledged", "name", "start_date", "end_date", "currency", "id"], axis=1, inplace=True)

        return data
    
    #reshape the 1-D country array to 2-D as fit_transform expects 2-D and finally fit the object 
    X = onehotencoder.fit_transform(data.Country.values.reshape(-1,1)).toarray()
    #To add this back into the original dataframe 
    dfOneHot = pd.DataFrame(X, columns = ["Country_"+str(int(i)) for i in range(data.shape[1])]) 
    df = pd.concat([data, dfOneHot], axis=1)
    #droping the country column 
    df= df.drop(['Country'], axis=1) 
    #printing to verify 
    print(df.head())

    def recodage(self, data):

        X = onehotencoder.fit_transform(data["country"].values.reshape(-1,1)).toarray()
        #To add this back into the original dataframe 
        dfOneHot = pd.DataFrame(X, columns = ["Country_"+str(int(i)) for i in range(data.shape[1])]) 
        data = pd.concat([data, data], axis=1)
        #droping the country column 
        data= data.drop(['Country'], axis=1) 
        feature_to_normalize =["goals_euro", "pledged_euro"]
        X = data[feature_to_normalize]
        X_norm = scaler.fit_transform(X)
        print(X_norm)



    def pretraitement(data):
        pass 


    def decoupage(self, X, y, test=False):
        # Split the data into train, validation and test dataset
        # 
        X_train, X_valid, y_train,y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_test, y_train,y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        if test:
            return X_test, y_test
        
        else:
            return X_train, X_valid, y_train, y_valid


    def main_runner():
        pass

preparation = DataPreparation(data)
data = preparation.nettoyage()
print(data)

preparation.recodage(data)