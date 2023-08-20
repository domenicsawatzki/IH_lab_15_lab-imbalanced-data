from .analysis_functions import *



def split_encode_use_logisticRegression(df):
    X = df[['tenure', 'SeniorCitizen', 'MonthlyCharges']]
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = split_the_data_into_train_test_datasets(X, y, 0.2)
    

    from sklearn.preprocessing import OneHotEncoder

    # Initialize the OneHotEncoder & Fit the encoder on the categorical data
    encoder = OneHotEncoder(drop='first')
    encoder.fit(X_train.select_dtypes(include= object))
    
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler().fit(X_train.select_dtypes(include= np.number))
    
    
    X_train_encoded = use_minMaxTransformer_and_oneHotEncoder(X_train, transformer, encoder)
    X_test_encoded = use_minMaxTransformer_and_oneHotEncoder(X_test, transformer, encoder)
    
    from sklearn.linear_model import LogisticRegression
    

    classification = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(X_train_encoded, y_train)
    
    predictions = classification.predict(X_test_encoded)
    
    score = classification.score(X_train_encoded, y_train)
    print(f'Model Score for Train = {score}')
    
    score = classification.score(X_test_encoded, y_test)
    print(f'Model Score for Test = {score}')
    
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, predictions))

    
    return predictions, y_train 
    

def discover_df(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_numeric = df.select_dtypes(include= np.number)
    corr_matrix = df_numeric.corr().round(2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(corr_matrix, annot=True)
    plt.show()
    
    hist_plot(df_numeric, 40)
    
def split_encode_use_neighbors(df):
    y = df['check_may']
    X = df.drop(['check_may'], axis=1)
    
    X_train, X_test, y_train, y_test = split_the_data_into_train_test_datasets(X, y, 0.2)
    

    from sklearn.preprocessing import OneHotEncoder

    # Initialize the OneHotEncoder & Fit the encoder on the categorical data
    encoder = OneHotEncoder(drop='first')
    encoder.fit(X_train.select_dtypes(include= object))
    
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler().fit(X_train.select_dtypes(include= np.number))
    
    
    X_train_encoded = use_minMaxTransformer_and_oneHotEncoder(X_train, transformer, encoder)
    X_test_encoded = use_minMaxTransformer_and_oneHotEncoder(X_test, transformer, encoder)
    
    from sklearn import neighbors
    

    classification = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
    classification.fit(X_train_encoded, y_train)
    
    predictions = classification.predict(X_test_encoded)
    
    score = classification.score(X_train_encoded, y_train)
    print(f'Model Score for Train = {score}')
    
    score = classification.score(X_test_encoded, y_test)
    print(f'Model Score for Test = {score}')
    
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, predictions))

    
    return predictions, y_train 