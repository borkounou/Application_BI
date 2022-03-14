
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
pd.options.mode.chained_assignment = None 


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
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


# Data path 
path = "../data/projects.csv"


# Pandas dataframe
data = pd.read_csv(path, encoding="ISO-8859-1")



class DataPreparation:
    '''
    Args:
        - data: Pandas dataframe
    '''
    def __init__(self, data):
        self.data = data
        self.instance_to_convert = ["sex", "category", "subcategory","state"]

    def currency_calculator(self, row_currency, row_goal):
        '''
        This function converts the different currency into euro: it is a currency converter, it converts the pledged and goal attributes
            Args:
                - row_currency(String): currency type of different nations
                - row_goal(Float): money to be converted

            Return 
                - euro(float): currency converted to Euro

        '''
        if row_currency== "USD":
            exchange =0.91
            euro = exchange * row_goal
            return euro
        elif row_currency == "CAD":
            exchange =0.71
            euro = exchange * row_goal
            return euro
        elif row_currency == "GBP":
            exchange =1.19
            euro = exchange * row_goal
            return euro

        elif row_currency == "AUD":
            exchange =0.66
            euro = exchange * row_goal
            return euro
        elif row_currency == "DKK":
            exchange =0.13
            euro = exchange * row_goal
            return euro
        elif row_currency == "SEK":
            exchange =0.09
            euro = exchange * row_goal
            return euro

        elif row_currency == "NOK":
            exchange =0.10
            euro = exchange * row_goal
            return euro
        elif row_currency =="NZD":
            exchange = 0.62
            euro = exchange * row_goal
            return euro
        elif row_currency == "CHF":
            exchange =0.97
            euro = exchange * row_goal
            return euro

        else:
            euro = row_goal
            return euro 
        

    def date_to_duration(self,start_date, end_date):
        '''
        Args:
            - start_date(String): starting date
            - end_date(String): Ending date

        Return:
            -absolute(duration.days)(int)
        '''

        start = datetime.datetime.strptime(start_date,'%Y-%m-%d %H:%M:%S' )
        end = datetime.datetime.strptime(end_date,'%Y-%m-%d %H:%M:%S' )
        duration = start -end
        return abs(duration.days)

    def class_binary(self,classe):
        if classe =="successful":
            return 1
        else:
            return 0


    def nettoyage(self):
        '''
        This function drops the nan values, convert the datetime into a duration, convert the different currencies into euro, and drops the attributes which are of no use

        return:
            - data: a pandas DataFrame
        '''
        # drop the nan values 
        print("Dropping the nan values...")
        data = self.data.dropna()
        data = data.sample(frac=1) # Shuffle the data
        data = data[:20000]
        # data = data[:1000]
        print("Dropping completed")
        # Subtract the start and end date in order to get the duration
        data["duration"] = data[["start_date", "end_date"]].apply(lambda x:self.date_to_duration(*x), axis=1)
        # Convert all the currency for goal attributes into Euro
        data['goal'] = data[["currency","goal"]].apply(lambda x:self.currency_calculator(*x),axis=1)
        # Convert all the currency for pledged attribute into Euro
        data['pledged'] = data[["currency","pledged"]].apply(lambda x:self.currency_calculator(*x), axis=1)

        # As drop the already used attributes or useless attributes such name, and Id
        data.drop(["start_date", "end_date","id", "currency", "name"], axis=1, inplace=True)

    
        return data
    

    def recodage(self, data, binairy=False, ignored_goal=False, ignored_pledged=False):
        '''
        Args:
            data: a pandas DataFrame
        return:
             - main_data: input data  
             - data["state"]: target data
        '''
        scaler = StandardScaler()
        if binairy:
            data["state"] = data.state.apply(lambda x:self.class_binary(x))
            if ignored_goal:
                data.drop(["goal"], axis=1, inplace=True)
            if ignored_pledged:
                data.drop(["pledged"], axis=1, inplace=True)
            data_numerical = data.select_dtypes(exclude='object')
            data_numerical_scaled = pd.DataFrame(scaler.fit_transform(data_numerical), columns=data_numerical.columns)
            print(data_numerical_scaled.columns)
            # data with categorical attributes
            data_categorical = data.select_dtypes(include='object')
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(data_categorical)
            X_cat = encoder.transform(data_categorical).toarray()
            # One hot encoded categorical data
            

            print(f"The following attributes have been one hot encoded: {data_categorical.columns.values}")
            data_categorical_encoded =pd.DataFrame(X_cat, columns= ["col"+str(i) for i in range(X_cat.shape[1])])
            main_data = pd.concat([data_numerical_scaled, data_categorical_encoded], axis=1)
            return main_data,data["state"], data_numerical_scaled, data_categorical_encoded



        else:
            # Separate the data  into the numerical and categorical
            # Normalization of data with numerical attributes
            if ignored_goal:
                data.drop(["goal"], axis=1, inplace=True)
            if ignored_pledged:
                data.drop(["pledged"], axis=1, inplace=True)
            data_numerical = data.select_dtypes(exclude='object')
            data_numerical_scaled = pd.DataFrame(scaler.fit_transform(data_numerical), columns=data_numerical.columns)
            print(data_numerical_scaled.columns)
            # data with categorical attributes
            data_categorical = data.select_dtypes(include='object').drop('state',axis=1)
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(data_categorical)
            X_cat = encoder.transform(data_categorical).toarray()
            # One hot encoded categorical data
            print(f"The following attributes have been one hot encoded: {data_categorical.columns.values}")
            data_categorical_encoded =pd.DataFrame(X_cat, columns= ["col"+str(i) for i in range(X_cat.shape[1])])

            # Merge the categorical and numerical data to form one big dataset

            main_data = pd.concat([data_numerical_scaled, data_categorical_encoded], axis=1)
    
            #convert the target class or the class to be predicted into numerical form
            unique_target =  data["state"].unique()
            print(f"The unique element of the target class {unique_target}")
            label_to_num = dict()
            for idx in range(len(unique_target)):
                label_to_num[unique_target[idx]] = idx

            data["state"] = data.state.apply(lambda x:label_to_num[x])
           
       
        return main_data,data["state"], data_numerical_scaled, data_categorical_encoded


    def pretraitement(self,data, categorical=False, numerical=False):
        '''
        Categorical:(boolean) if set to true return the already one encoded categorical data for some specific classifier such as naive bayes classifier
        '''
        main_data,  target, data_numerical_scaled, data_categorical_encoded =self.recodage(data, binairy=True, ignored_goal=True)
        if categorical:
            return  data_categorical_encoded,target
        elif numerical:
            return data_numerical_scaled,target
        else:
            return main_data, target
         

    def decoupage(self, data, categorical_data =False, numerical_data=False):
        '''
        Args:
            - X(pandas DataFrame): input data
            - y(pandas DataFrame): target data
        
        Return:
            - X_train
            - X_valid
            - X_test
            - y_train 
            - y_valid
            - y_test

        '''

        if categorical_data:
            data_categorical, target = self.pretraitement(data, categorical=categorical_data)
            # Split Train data and validation
            X_train, X_valid, y_train,y_valid = train_test_split(data_categorical, target, test_size=0.1, random_state=42)
            # 10 % of test data and 10% of valid data and 80 % of train data
            X_train, X_test, y_train,y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            return X_train, X_valid,X_test, y_train, y_valid,y_test

        elif numerical_data:
            data_numerical_scaled, target = self.pretraitement(data, numerical=numerical_data)
            X_train, X_valid, y_train,y_valid = train_test_split(data_numerical_scaled, target, test_size=0.1, random_state=42)
            # 10 % of test data and 10% of valid data and 80 % of train data
            X_train, X_test, y_train,y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            return X_train, X_valid,X_test, y_train, y_valid,y_test

        
        else:
            main_data, target = self.pretraitement(data)
            # Split the data into train, validation and test dataset 
            # 10 % of validation data
            X_train, X_valid, y_train,y_valid = train_test_split(main_data, target, test_size=0.1, random_state=42)
            # 10 % of test data and 10% of valid data and 80 % of train data
            X_train, X_test, y_train,y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

            return X_train, X_valid,X_test, y_train, y_valid,y_test


def main_data_splitter(data, categorical = False, numerical=False):
    '''
    Args: 
        - data: pandas DataFrame
    Return:
    - X_train
    - X_valid
    - X_test
    - y_train 
    - y_valid
    - y_test
    '''

    preparation = DataPreparation(data)
    data = preparation.nettoyage()
    if categorical:
        X_train, X_valid,X_test, y_train, y_valid,y_test = preparation.decoupage(data, categorical_data=categorical)
        print(f'Shape of Xtrain {X_train.shape}')
        print(f'Shape of XValid {X_valid.shape}')
        print(f'Shape of Xtest {X_test.shape}')
        print(f'Shape of ytrain {y_train.shape}')
        print(f'Shape of yvalid {y_valid.shape}')
        print(f'Shape of ytest {y_test.shape}')
        return X_train, X_valid,X_test, y_train, y_valid,y_test
    
    elif numerical:
        X_train, X_valid,X_test, y_train, y_valid,y_test = preparation.decoupage(data, numerical_data=numerical)
        print(f'Shape of Xtrain {X_train.shape}')
        print(f'Shape of XValid {X_valid.shape}')
        print(f'Shape of Xtest {X_test.shape}')
        print(f'Shape of ytrain {y_train.shape}')
        print(f'Shape of yvalid {y_valid.shape}')
        print(f'Shape of ytest {y_test.shape}')
        return X_train, X_valid,X_test, y_train, y_valid,y_test


    else:
        X_train, X_valid,X_test, y_train, y_valid,y_test = preparation.decoupage(data)
        print(f'Shape of Xtrain {X_train.shape}')
        print(f'Shape of XValid {X_valid.shape}')
        print(f'Shape of Xtest {X_test.shape}')
        print(f'Shape of ytrain {y_train.shape}')
        print(f'Shape of yvalid {y_valid.shape}')
        print(f'Shape of ytest {y_test.shape}')

        return X_train, X_valid,X_test, y_train, y_valid,y_test




X_train, X_valid,X_test, y_train, y_valid,y_test= main_data_splitter(data)
#X_train, X_valid,X_test, y_train, y_valid,y_test= main_data_splitter(data, numerical=True)
# X_train, X_valid,X_test, y_train, y_valid,y_test= main_data_splitter(data, categorical=True)

# print(X_train)
