"""STEP 1: preliminary language-specific commands"""
import numpy as np
import pandas as pd
from sklearn import model_selection, metrics, datasets
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

train_score=0
test_score=0
error_score_train=[];
error_score_test=[];
hp=[];
mse1=np.array([]);
hp1=np.array([]);
ch1=input("Enter R for regression and C for classification : ");
if ch1== "R":
    method=input("Enter LR for Regression\t,DT for Decision Tree, KNN for KNeighbour, RF for Random Forest, ANN for Artificial Neural Network :")
    iterations=input("Enter the number of iterations :")
    iterations=int(iterations)
    """STEP 2: load the data"""
    dataset=input("Enter db for diabetes and lr for linnerud cl for housing :")
    if dataset=="db":
        input_data, output_data = datasets.load_diabetes(return_X_y=True)
    elif dataset=="lr":
        input_data, output_data = datasets.load_linnerud(return_X_y=True)
    elif dataset=='cl':
        input_data, output_data = datasets.fetch_california_housing(return_X_y=True);
    else :
        print("Invalid selection default dataset diabetes loaded")
        input_data, output_data = datasets.load_diabetes(return_X_y=True)


    """STEP 3: shuffle the samples and split into train and test"""
    ##features=input("Enter the value of K,features for selectBest :");
    #features=int(features);
    choice_cluster=input("Enter the number of clusters for Kmeans :");
    choice_cluster=int(choice_cluster);
    pca = PCA(n_components=4, svd_solver='full')
    pca.fit(input_data);
    input_data=pca.fit_transform(input_data, y=None)
    kmeans = KMeans(n_clusters= choice_cluster)
    for n in range(1,5):
        for x in range(0,iterations): #iterations
            [train_in, test_in, train_out, test_out] = model_selection.train_test_split(input_data, output_data, test_size=.30,random_state=42)
            """Scaling Data """
            
            scaler = StandardScaler();
            train_in=scaler.fit_transform(train_in);
            test_in=scaler.transform(test_in);
            """ PCA implementation"""
            train_in=pca.fit_transform(train_in,y=None);
            test_in=pca.fit_transform(test_in,y=None);
            """K Best Implementation"""
            train_in = SelectKBest(k='all').fit_transform(train_in, train_out);
            test_in = SelectKBest(k='all').fit_transform(test_in, test_out);
            """K-Means implementation"""
            kmeans.fit(train_in);
            train_clusters=kmeans.labels_
            kmeans.fit(test_in);
            test_clusters=kmeans.labels_
            train_in = np.concatenate((train_in, np.reshape(train_clusters, (train_clusters.size, 1))), axis = 1)
            test_in = np.concatenate((test_in, np.reshape(test_clusters, (test_clusters.size, 1))), axis = 1)
            
            """STEP 4: determine the hyperparameters"""
            if method == "LR":
                model = LinearRegression(fit_intercept = True)
            elif method == "DT":
                model = DecisionTreeRegressor(max_depth=30, min_samples_split=10, min_samples_leaf=10)
            elif method == "KNN":
                model = KNeighborsRegressor(n_neighbors=n,weights='distance',metric='manhattan')
            elif method=="RF":
                model= RandomForestRegressor(n_estimators=100,min_samples_split=2, min_samples_leaf=1,max_depth=2,bootstrap=True)
            elif method=="ANN":
                model= MLPRegressor(hidden_layer_sizes=n,solver='sgd',alpha=0.01,activation='logistic')
            else :
                print("Please enter a valid method selection");
                break;

            """STEP 5: train the model"""
            model.fit(train_in, train_out)
            
            """STEP 6: predict training outputs"""
            pred_train_out = model.predict(train_in)
            
            """STEP 7: evaluate the training data"""
            eval_method = 'mean squared error'
            train_score += metrics.mean_squared_error(train_out, pred_train_out)
            
            """STEP 8: predict test outputs"""
            pred_test_out = model.predict(test_in)
                
            """STEP 9: get the testing score"""
            test_score += metrics.mean_squared_error(test_out, pred_test_out);
        train_score=train_score/iterations
        test_score=test_score/iterations
        error_score_train.append(train_score);
        error_score_test.append(test_score);
        hp.append(n);
else:
    dataset=input("Enter ir for iris, dg for digits and wn for wine :")
    if dataset=="ir":
        input_data, output_data = datasets.load_iris(return_X_y=True)
    elif dataset=="dg":
        input_data, output_data = datasets.load_digits(return_X_y=True)
    elif dataset=='wn':
        input_data, output_data = datasets.load_wine(return_X_y=True);
    else :
        print("Invalid selection default dataset iris loaded")
        input_data, output_data = datasets.load_iris(return_X_y=True)
    ch2=input("Enter S for SVM , L for logistic, N for nearest neighbors , D for Decision Tree, R for RandomForestClassifier and A for ANN  : ")
    [train_in, test_in, train_out, test_out] = model_selection.train_test_split(input_data, output_data, test_size=.30,random_state=42)
    scaler = StandardScaler();
    train_in=scaler.fit_transform(train_in);
    test_in=scaler.transform(test_in);
    for k in range(0,10):
        if ch2=="S":
            #ch3=input('Enter the Kernel that you want to us kernel ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ :')
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel='linear'))
        elif ch2=="L":
            #ch4= input('Enter the value of solver ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ :  ')
            clf = LogisticRegression(random_state=0,solver='lbfgs').fit(train_in, train_out)
        elif ch2=="N":
           # ch5=input('Enter the value of n for n_neighbors : ')
            clf = KNeighborsClassifier(n_neighbors=7,weights='distance',metric='euclidean')
        elif ch2=="D":
            clf = DecisionTreeClassifier(random_state=None,criterion='entropy',max_depth=10,min_samples_split=5,min_samples_leaf=1)
        elif ch2=="R":
            clf = RandomForestClassifier(random_state=None,n_estimators=10,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,bootstrap=True)
        elif ch2=="A":
            clf = MLPClassifier(random_state=None, max_iter=300,hidden_layer_sizes=10,solver='sgd',alpha=0.01,activation='tanh')
        else :
            print("Default method Logistic Regression selected");
            clf = LogisticRegression(random_state=0,solver='lbfgs').fit(train_in, train_out)
            
        clf.fit(train_in, train_out)
        pred_out=clf.predict(test_in)
        
    
    """STEP 10: save evaluation results to a file"""
    # error_score_train=[train_score];
    # error_score_test=[test_score];
""" Loop to run iterations"""


"""Average of train and test score"""


"""STEP 11: display results to the console"""
if ch1== "R":
    print('Method Used '+method)
    print('training ' + eval_method + ' (%)')
    print(train_score)
    print('testing ' + eval_method + ' (%)')
    print(test_score)
    print(error_score_train);
    print(error_score_test);
    plt.plot(hp, error_score_train);
    plt.xlabel("Value of Hyperparameter")
    plt.ylabel("Error Value in %")
    plt.show()
    plt.plot(hp, error_score_test);
    plt.xlabel("Value of Hyperparameter")
    plt.ylabel("Error Value in %")
    plt.show()
    pd.DataFrame(np.array([train_score, test_score])).to_csv(eval_method + "4.csv", index = False, header = False)
else:
    print(classification_report(test_out, pred_out)) 
    
    