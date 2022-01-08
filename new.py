import streamlit as st

# import streamlit as st
import pandas as pd
import pandas_profiling as pp
import time
import datetime
# import module
import webbrowser


import numpy as np
from scikitplot.metrics import plot_confusion_matrix
# from textblob import TextBlob
import seaborn as sns
# from nlp import NLPNaive

from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler       # Preprocessing method
from sklearn.neighbors import KNeighborsClassifier   # K-Nearest Neighbours Classifier model
from sklearn.svm import SVC                          # Support Vector Classifier model
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier model




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics


# Create space betwwen two context
def space():
    st.markdown("<br>", unsafe_allow_html=True)


# Notebook Contains:-
# 1.Import all the neceassary modules and load the data
#
# 2.Exploratory Data Analysis
#
# 3.Preprocessing
#
# 3.Feature Selection
#
# 4.Feature Scaling
#
# 5.Test - Train Split
#
# 6.Modelling & Evaluation
# dew
# Heading
st.markdown("<h1 style='text-align: center; color: #3f3f44'>AUTO ML</h1>", unsafe_allow_html=True)
space()
space()
# Sub-Heading
st.markdown("<strong><p style='color: #424874'> This project is Automated machine learning,which is also referred to as automated ML or AutoML, is the process of automating the time-consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality.</p></strong>",
            unsafe_allow_html=True)
st.markdown("<strong><h2 style='color: #424874'>AUTO ML Contains following step:-</h2></strong>",
            unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>1.load the data</p></strong>",
            unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>2.Exploratory Data Analysis</p></strong>",
            unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>3.Feature Selection</p></strong>",
            unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>4.selecting algorithm</p></strong>",
            unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>5.Model Evaluation</p></strong>",
            unsafe_allow_html=True)


# st.markdown(
#     "<strong><h4 style='color: #424874'>2) You can choose different cleaning process (Stemming, Lemmatizing)</h4></strong>",
#     unsafe_allow_html=True)
# st.markdown(
#     "<strong><p style='color: #424874'>3) Different type of  Metrics formation (Count Vectorizing, TF-IDF)</p></strong>",
#     unsafe_allow_html=True)
# st.markdown(
#     "<strong><p style='color: #424874'>4) Plotting Sentimental Analysis, Confusion Metrics and Word Cloud</p></strong>",
#     unsafe_allow_html=True)
space()

# usecase = st.sidebar.selectbox("Usecase",["Machine Learning","NLP"]),
# if usecase == "Machine Learning":
#     # write the selected options
#     st.write("You selected", len(usecase), 'hobbies')
#     # ML_model = st.sidebar.radio("Usecase",["Classification","Regression"]),
#     # if ML_model == "Classification":
#     #     ML_algo_classification = st.sidebar.selectbox(
#     #         'How would you like to be contacted?',
#     #         ('logistic regression', 'decision tree', 'Random Forest','SVM'))
#     # elif ML_model == "Regression":
#     #     ML_algo_regression = st.sidebar.selectbox(
#     #         'How would you like to be contacted?',
#     #         ('linear regression', 'decision tree regressor', 'Random Forest regressor', 'SVR'))
#
# elif usecase == "NLP":
#     print("hello NLP")

# t.sliderbar.radio("hello",["home","ABOUT us"]
# st.sidebar.multiselect("slect a column",data)
# st.sidebar:
#     user_word = st.text_input("Enter a keyword", "habs")
#     select_language = st.radio('Tweet language', ('All', 'English', 'French'))
#     include_retweets = st.checkbox('Include retweets in data')
#     num_of_tweets = st.number_input('Maximum number of tweets', 100)
#     submitted1 = st.form_submit_button(label = 'Search Twitter 🔎')



# Main function
# st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
# space()
#
# st.markdown(
#     "<strong><h3 style='color: #424874'> Accuracy of Linear Regresion model</h3></strong>",
#     unsafe_allow_html=True)
# st.markdown(
#     "<strong><h3 style='color: #424874'> Different Errors estimators of Linear Regresion model</h3></strong>",
#     unsafe_allow_html=True)
# st.markdown("<h4 style='text-align: center; color: #3f3f44'>Prediciton Graph (prediction vs y_test) of Linear Regresion model</h4>", unsafe_allow_html=True)
#
# st.markdown(
#             "<strong><h3 style='color: #424874'>2 Download Pickle file </h3></strong>",
#             unsafe_allow_html=True)

def app():
    def linear_regression(X, Y):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        regr = LinearRegression()
        regr.fit(X_train, Y_train)

        st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Accuracy of Linear Regresion model</h3></strong>",
            unsafe_allow_html=True)
        print("accuracy is : ", regr.score(X_test, Y_test))
        st.write("accuracy is : ", regr.score(X_test, Y_test))

        pred = regr.predict(X_test)

        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Different Errors estimators to evaluate Linear Regresion model</h3></strong>",
            unsafe_allow_html=True)
        st.write("MAE", metrics.mean_absolute_error(Y_test, pred))
        st.write('MSE:', metrics.mean_squared_error(Y_test, pred))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, pred)))
        st.write('R_sqaure:', metrics.r2_score(Y_test, pred))
        print('MAE:', metrics.mean_absolute_error(Y_test, pred))
        print('MSE:', metrics.mean_squared_error(Y_test, pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, pred)))
        print('R_sqaure:', metrics.r2_score(Y_test, pred))

        space()
        # plt.scatter(Y_test, pred, alpha=0.5)

        st.markdown("<h4 style='text-align: center; color: #3f3f44'>Prediciton Graph (prediction vs y_test) of Linear Regresion model</h4>", unsafe_allow_html=True)
        plt.xlabel("Y_test")
        plt.ylabel("pred")
        fig = plt.figure(figsize=(10, 4))
        plt.scatter(Y_test,pred)

        # st.balloons('Done')
        st.balloons()
        st.pyplot(fig)
        # plt.show()
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'>Download Pickle file </h3></strong>",
            unsafe_allow_html=True)
        import pickle
        # open a file, where you ant to store the data
        pickle_file = open('linear_regression.pkl', 'wb')
        pickle.dump(regr, pickle_file)
        with open("linear_regression.pkl", "rb") as file:
            st.download_button(
                label="Pickle file ",
                data=file,
                file_name="linear_regression.pkl"
            )
        return

    def desicion_tree_regressor(X, Y):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        DT = DecisionTreeRegressor()
        DT.fit(X_train, Y_train)

        st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Accuracy of desicion tree regressor model</h3></strong>",
            unsafe_allow_html=True)

        score = DT.score(X_train, Y_train)
        print(score)
        st.write("accuracy is : ", DT.score(X_test, Y_test))
        pred = DT.predict(X_test)

        space()
        st.markdown(
            "<strong><h3 style='color: #424874'> Different Errors estimators to evaluate desicion tree regressor model</h3></strong>",
            unsafe_allow_html=True)
        st.write("MAE", metrics.mean_absolute_error(Y_test, pred))
        st.write('MAE:', metrics.mean_absolute_error(Y_test, pred))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, pred)))
        st.write('R_sqaure:', metrics.r2_score(Y_test, pred))

        print('MAE:', metrics.mean_absolute_error(Y_test, pred))
        print('MSE:', metrics.mean_squared_error(Y_test, pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, pred)))
        print('R_sqaure:', metrics.r2_score(Y_test, pred))

        space()
        # plt.scatter(Y_test, pred, alpha=0.5)

        st.markdown(
            "<h4 style='text-align: center; color: #3f3f44'>Prediciton Graph (prediction vs y_test) of desicion tree regressor model</h4>",
            unsafe_allow_html=True)
        plt.xlabel("Y_test")
        plt.ylabel("pred")
        fig = plt.figure(figsize=(10, 4))
        plt.scatter(Y_test, pred)

        st.balloons()
        st.pyplot(fig)

        space()

        st.markdown(
            "<strong><h3 style='color: #424874'>Download Pickle file </h3></strong>",
            unsafe_allow_html=True)
        import pickle
        # open a file, where you ant to store the data
        pickle_file = open('desicion_tree_regressor.pkl', 'wb')
        pickle.dump(DT, pickle_file)
        with open("desicion_tree_regressor.pkl", "rb") as file:
            st.download_button(
                label="Download Pickle file ",
                data=file,
                file_name="desicion_tree_regressor.pkl"
            )
        return

    def random_forest_regressor(X, Y):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        RF = RandomForestRegressor()
        RF.fit(X_train, Y_train)

        st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Accuracy of random forest regressor model</h3></strong>",
            unsafe_allow_html=True)


        #
        # st.markdown(
        #             "<strong><h3 style='color: #424874'>2 Download Pickle file </h3></strong>",
        #             unsafe_allow_html=True)

        train_score = RF.score(X_train, Y_train)
        st.write("training accuracy is : ", RF.score(X_test, Y_test))
        print(train_score)
        test_score = RF.score(X_test, Y_test)
        st.write("test accuracy is : ", RF.score(X_test, Y_test))
        print(test_score)

        pred = RF.predict(X_test)

        space()
        st.markdown(
            "<strong><h3 style='color: #424874'> Different Errors estimators to evaluate random forest regressor model</h3></strong>",
            unsafe_allow_html=True)
        st.write("MAE", metrics.mean_absolute_error(Y_test, pred))
        st.write('MAE:', metrics.mean_absolute_error(Y_test, pred))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, pred)))
        st.write('R_sqaure:', metrics.r2_score(Y_test, pred))

        print('MAE:', metrics.mean_absolute_error(Y_test, pred))
        print('MSE:', metrics.mean_squared_error(Y_test, pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, pred)))
        print('R_sqaure:', metrics.r2_score(Y_test, pred))

        space()

        st.markdown("<h4 style='text-align: center; color: #3f3f44'>Prediciton Graph (prediction vs y_test) of random forest regressor model</h4>", unsafe_allow_html=True)
        # plt.scatter(Y_test, pred, alpha=0.5)
        plt.xlabel("Y_test")
        plt.ylabel("pred")
        fig = plt.figure(figsize=(10, 4))
        plt.scatter(Y_test, pred)

        st.balloons()
        st.pyplot(fig)

        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Download Pickle file </h3></strong>",
            unsafe_allow_html=True)
        import pickle
        # open a file, where you ant to store the data
        pickle_file = open('random_forest_regressor.pkl', 'wb')
        pickle.dump(RF, pickle_file)
        with open("random_forest_regressor.pkl", "rb") as file:
            st.download_button(
                label="Download Pickle file ",
                data=file,
                file_name="random_forest_regressor.pkl"
            )
        return


    def logistic_regression(X, Y):
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        LR = LogisticRegression(random_state=0)
        LR.fit(X_train, Y_train)

        st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Accuracy of logistic Regression</h3></strong>",
            unsafe_allow_html=True)
        train_score = LR.score(X_train, Y_train)
        st.write("training accuracy is : ", LR.score(X_test, Y_test))
        print(train_score)
        test_score = LR.score(X_test, Y_test)
        st.write("test accuracy is : ", LR.score(X_test, Y_test))
        print(test_score)

        LR_pred = LR.predict(X_test)

        space()
        st.markdown(
            "<h4 style='text-align: center; color: #3f3f44'>Confusion metrix of logistic regresssion model</h4>",
            unsafe_allow_html=True)
        cnf_matrix_log = confusion_matrix(Y_test, LR_pred)
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True, cmap="Greys", fmt='g', ax=ax)
        st.write(fig)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Different estimators to evaluate logistic regression model</h3></strong>",
            unsafe_allow_html=True)
        st.write("Accuracy of Logistic regresion", accuracy_score(Y_test, LR_pred))
        st.write("f1 score of this logistic regression", f1_score(Y_test, LR_pred))
        st.write("recall score of logistic regression", recall_score(Y_test, LR_pred))
        st.write("precision score of logistic regression", precision_score(Y_test, LR_pred))

        print("Accuracy of Logistic regresion", accuracy_score(Y_test, LR_pred))
        print("f1 score of this logistic regression", f1_score(Y_test, LR_pred))
        print("recall score of logistic regression", recall_score(Y_test, LR_pred))
        print("precision score of logistic regression", precision_score(Y_test, LR_pred))

        # st.subheader(“Confusion Matrix”)

        from sklearn.metrics import roc_auc_score

        # figure= plt.subplots()
        # auc = np.round(roc_auc_score(Y_test, LR_pred), 3)
        # str.write(auc)
        # print("Report : ", classification_report(Y_test, LR_pred))
        # # st.write("Report : ", classification_report(Y_test, LR_pred))
        # report = classification_report(Y_test, LR_pred)
        # st.write(report)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Download Pickle file </h3></strong>",
            unsafe_allow_html=True)
        import pickle
        # open a file, where you ant to store the data
        pickle_file = open('logistic_regression.pkl', 'wb')
        pickle.dump(LR, pickle_file)
        with open("logistic_regression.pkl", "rb") as file:
            st.download_button(
                label="Download Pickle file ",
                data=file,
                file_name="logistic_regression.pkl"
            )
        st.balloons()
        return

    def decision_tree(X, Y):

        from sklearn.tree import DecisionTreeClassifier
        from sklearn import tree

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
        # DT_classifier=DecisionTreeClassifier(criterion='gini',max_depth=11,max_features=6,max_leaf_nodes=6)
        DT_classifier = DecisionTreeClassifier()
        DT_classifier.fit(X_train, Y_train)

        st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Accuracy of Decision tree classifier</h3></strong>",
            unsafe_allow_html=True)
        train_score = DT_classifier.score(X_train, Y_train)
        st.write("training accuracy is : ",DT_classifier.score(X_train, Y_train))
        print(train_score)
        test_score = DT_classifier.score(X_test, Y_test)
        st.write("test accuracy is : ",DT_classifier.score(X_test, Y_test))
        print(test_score)

        DT_pred = DT_classifier.predict(X_test)

        space()
        st.markdown(
            "<h4 style='text-align: center; color: #3f3f44'>Confusion Matrix Decision tree classifier model</h4>",
            unsafe_allow_html=True)
        cnf_matrix_log = confusion_matrix(Y_test, DT_pred)
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True, cmap="Blues", fmt='g', ax=ax)
        st.write(fig)

        space()
        st.markdown(
            "<strong><h3 style='color: #424874'> Different estimators to evaluate Decision tree model</h3></strong>",
            unsafe_allow_html=True)
        st.write("Accuracy of Decision tree", accuracy_score(Y_test, DT_pred))
        st.write("f1 score of this Decision tree", f1_score(Y_test, DT_pred))
        st.write("recall score of Decision tree", recall_score(Y_test, DT_pred))
        st.write("precision score of Decision tree", precision_score(Y_test, DT_pred))

        print("Accuracy of Decision tree", accuracy_score(Y_test, DT_pred))
        print("f1 score of this Decision tree", f1_score(Y_test, DT_pred))
        print("recall score of Decision tree", recall_score(Y_test, DT_pred))
        print("precision score of Decision tree", precision_score(Y_test, DT_pred))

        # cnf_matrix_log = confusion_matrix(Y_test, DT_pred)
        #
        # sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True, cmap="Greens", fmt='g')
        # # ax.xaxis.set_label_position("top")
        # plt.tight_layout()
        # plt.title('Confusion matrix Decision tree\n', y=1.1)'
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Download Pickle file </h3></strong>",
            unsafe_allow_html=True)
        import pickle
        # open a file, where you ant to store the data
        pickle_file = open('Decision_tree.pkl', 'wb')
        pickle.dump(DT_classifier, pickle_file)
        with open("Decision_tree.pkl", "rb") as file:
            st.download_button(
                label="Download Pickle file ",
                data=file,
                file_name="Decision_tree.pkl"
            )

        print("Report : ", classification_report(Y_test, DT_pred))
        st.balloons()
        return

    def random_forest(X, Y):
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        RF_clf = RandomForestClassifier(n_estimators=100, max_depth=15)
        RF_clf.fit(X_train, Y_train)

        st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Accuracy of random forest classifier</h3></strong>",
            unsafe_allow_html=True)
        train_score = RF_clf.score(X_train, Y_train)
        st.write("training accuracy is : ", RF_clf.score(X_train, Y_train))
        print(train_score)
        test_score = RF_clf.score(X_test, Y_test)
        st.write("test accuracy is : ", RF_clf.score(X_test, Y_test))
        print(test_score)

        RF_pred = RF_clf.predict(X_test)

        space()
        st.markdown(
            "<h4 style='text-align: center; color: #3f3f44'>Confusion Matrix Random Forest classifier model</h4>",
            unsafe_allow_html=True)
        cnf_matrix_log = confusion_matrix(Y_test, RF_pred)
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True, cmap='Purples', fmt='g', ax=ax)
        st.write(fig)

        space()
        st.markdown(
            "<strong><h3 style='color: #424874'> Different estimators to evaluate Random forest model</h3></strong>",
            unsafe_allow_html=True)
        st.write("Accuracy of Random forest", accuracy_score(Y_test, RF_pred))
        st.write("f1 score of this Random forest", f1_score(Y_test, RF_pred))
        st.write("recall score of Random forest", recall_score(Y_test, RF_pred))
        st.write("precision score of Random forest", precision_score(Y_test, RF_pred))

        print("Accuracy of Random forest", accuracy_score(Y_test, RF_pred))
        print("f1 score of this random forest", f1_score(Y_test, RF_pred))
        print("recall score of random forest", recall_score(Y_test, RF_pred))
        print("precision score of random forest", precision_score(Y_test, RF_pred))

        # cnf_matrix_log = confusion_matrix(Y_test, RF_pred)
        #
        # sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True, cmap="Greens", fmt='g')
        # # ax.xaxis.set_label_position("top")
        # plt.tight_layout()
        # plt.title('Confusion matrix Random forest\n', y=1.1)

        print("Report : ", classification_report(Y_test, RF_pred))

        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Download Pickle file </h3></strong>",
            unsafe_allow_html=True)
        import pickle
        # open a file, where you ant to store the data
        pickle_file = open('Random_forest.pkl', 'wb')
        pickle.dump(RF_clf, pickle_file)
        with open("Random_forest.pkl", "rb") as file:
            st.download_button(
                label="Download Pickle file ",
                data=file,
                file_name="Random_forest.pkl"
            )
        st.balloons()

        return

    def KNN(X, Y):
        from sklearn.neighbors import KNeighborsClassifier

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        Knn = KNeighborsClassifier(n_neighbors=5)
        Knn.fit(X_train, Y_train)

        st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
        space()

        st.markdown(
            "<strong><h3 style='color: #424874'> Accuracy of KNN classifier</h3></strong>",
            unsafe_allow_html=True)

        train_score = Knn.score(X_train, Y_train)
        st.write("training accuracy is : ", Knn.score(X_train, Y_train))
        print(train_score)
        test_score = Knn.score(X_test, Y_test)
        st.write("test accuracy is : ", Knn.score(X_test, Y_test))
        print(test_score)

        Knn_pred = Knn.predict(X_test)

        space()
        st.markdown(
            "<h4 style='text-align: center; color: #3f3f44'>Confusion Matrix KNN classifier model</h4>",
            unsafe_allow_html=True)

        cnf_matrix_log = confusion_matrix(Y_test, Knn_pred)
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True, cmap="Oranges", fmt='g', ax=ax)
        st.write(fig)

        space()
        st.markdown(
            "<strong><h3 style='color: #424874'> Different estimators to evaluate KNN model</h3></strong>",
            unsafe_allow_html=True)
        st.write("Accuracy of KNN", accuracy_score(Y_test, Knn_pred))
        st.write("f1 score of this KNN", f1_score(Y_test, Knn_pred))
        st.write("recall score of KNN", recall_score(Y_test, Knn_pred))
        st.write("precision score of KNN", precision_score(Y_test, Knn_pred))

        print("Accuracy of KNN", accuracy_score(Y_test, Knn_pred))
        print("f1 score of this KNN", f1_score(Y_test, Knn_pred))
        print("recall score of  KNN", recall_score(Y_test, Knn_pred))
        print("precision score of KNN", precision_score(Y_test, Knn_pred))

        print("Report : ", classification_report(Y_test, Knn_pred))

        space()

        # st.markdown(
        #     "<strong><h3 style='color: #424874'> Download Pickle file </h3></strong>",
        #     unsafe_allow_html=True)

        st.balloons()
        return

    def SVC(X, Y):

        from sklearn.svm import SVC
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
            clf = SVC()
            clf.fit(X_train, Y_train)
        except Exception as e:
            print(" ERROR in fitting or splitting SVC model : ", e)

        try:
            st.markdown("<h2 style='text-align: center; color: #3f3f44'>Predicition</h2>", unsafe_allow_html=True)
            space()
            st.markdown("<strong><h3 style='color: #424874'> Accuracy of SVM</h3></strong>",unsafe_allow_html=True)
            train_score = clf.score(X_train, Y_train)
            st.write("training accuracy is : ", clf.score(X_train, Y_train))
            print(train_score)
            test_score = clf.score(X_test, Y_test)
            st.write("test accuracy is : ", clf.score(X_test, Y_test))
            print(test_score)
            clf_pred = clf.predict(X_test)
            space()
        except Exception as e:
            print(" ERROR in Accuray / Score SVC model : ", e)

        try:
            st.markdown( "<h4 style='text-align: center; color: #3f3f44'>Confusion Matrix SVM classifier model</h4>",unsafe_allow_html=True)
            cnf_matrix_log = confusion_matrix(Y_test, clf_pred)
            fig, ax = plt.subplots()
            sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True, cmap="Greens", fmt='g', ax=ax)
            st.write(fig)
            space()
        except Exception as e:
            print(" ERROR in Accuray / Score SVC model : ", e)

        try:
            st.markdown("<strong><h3 style='color: #424874'> Different estimators to evaluate SVM model</h3></strong>",unsafe_allow_html=True)
            st.write("Accuracy of Logistic regresion", accuracy_score(Y_test, clf_pred))
            st.write("f1 score of this logistic regression", f1_score(Y_test, clf_pred))
            st.write("recall score of logistic regression", recall_score(Y_test, clf_pred))
            st.write("precision score of logistic regression", precision_score(Y_test, clf_pred))

            print("Accuracy of svc", accuracy_score(Y_test, clf_pred))
            print("f1 score of this svc", f1_score(Y_test, clf_pred))
            print("recall score of  svc", recall_score(Y_test, clf_pred))
            print("precision score of svc", precision_score(Y_test, clf_pred))
            space()
        except Exception as e:
            print(" ERROR in classification report SVC model : ", e)

        try:
            st.markdown("<strong><h3 style='color: #424874'> Download Pickle file </h3></strong>",unsafe_allow_html=True)
            import pickle
            # open a file, where you ant to store the data
            pickle_file = open('svc.pkl', 'wb')
            pickle.dump(SVC, pickle_file)
            with open("svc.pkl", "rb") as file:
                st.download_button(
                    label="Download Pickle file ",
                    data=file,
                    file_name="svc.pkl"
                )
            st.balloons()
        except Exception as e:
            print(" ERROR in pickle file of SVC model : ", e)
        return


    st.markdown(
        "<strong><h3 style='color: #424874'>Step1) Upload data (csv, txt, xlsx) </h3></strong>",
        unsafe_allow_html=True)
    df = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
    space()

    if df is not None:
        # Reading data

        data = pd.read_csv(df, encoding="ISO-8859-1")
        st.dataframe(data.head())
        space()

        usecase = st.selectbox("Select your Usecase", ["Machine Learning", "NLP"])
        if usecase == "Machine Learning":
            st.write("You selected", usecase)
            ML_model = st.radio("select your model", ["Classification", "Regression"])
            st.write("You selected", ML_model)

            st.markdown(
                "<strong><h3 style='color: #424874'>Step2) Performing Explorator data analaysis</h3></strong>",
                unsafe_allow_html=True)
            if st.button("EDA"):
                space()
                profile = pp.ProfileReport(data)
                profile.to_file("output.html")
                webbrowser.open('output.html')


                with open("output.html", "rb") as file:
                    st.download_button(
                        label="Download EDA Report",
                        data=file,
                        file_name="output.html"
                    )
            space()
            try:
                st.markdown(
                    "<strong><h3 style='color: #424874'>Step3) Feature Selection</h3></strong>",
                    unsafe_allow_html=True)
                X_label = st.multiselect("Select Text Column", data.columns)
                text = X_label
                print(text)
                space()
                target = st.selectbox("Select Target Column", data.columns)
                print(target)
                space()
                print(type(text), type(target))
                #
                text.append(target)
                # # Reassigning feature to DataFrame
                data = data[text]
                print(data)

                # # Droping NaN values
                data = data.dropna()

                data[X_label].dropna()
                x =data[X_label]
                print(x)

                data[target].dropna()
                y =data[target]
                print(y)
            except Exception as e:
                st.write(" Please Select valid Input and output feature for prediction: ", e) 

            st.markdown("<h4 style='color: #438a5e'>Final Dataset</h4>", unsafe_allow_html=True)
            st.dataframe(data.head())
            space()

            space()

            st.markdown(
                "<strong><h3 style='color: #424874'>Step4) Model selection </h3></strong>",
                unsafe_allow_html=True)
            if ML_model == "Classification":
                ML_algo_classification = st.selectbox(
                    'How would you like to be contacted?',
                    ('logistic regression', 'decision tree', 'Random Forest', 'KNN', 'SVC'))
                st.write("You selected", ML_algo_classification)
            elif ML_model == "Regression":
                ML_algo_regression = st.selectbox(
                    'How would you like to be contacted?',
                    ('linear regression', 'decision tree regressor', 'Random Forest regressor'))
                print(ML_algo_regression)

        elif usecase == "NLP":
            st.info('Sorry!! this feature is still in progress, Wait till next release ,It will be available soon')

        # Model Creation
        try:
            
            if ML_model == 'Regression':
                if st.button("Analyze"):
                    space()
                    try:
                        if ML_algo_regression == 'linear regression':
                            linear_regression(x,y)
                        elif ML_algo_regression == 'decision tree regressor':
                            desicion_tree_regressor(x,y)
                        elif ML_algo_regression == 'Random Forest regressor':
                            random_forest_regressor(x,y)
                    except Exception as e:
                        st.warning(" Please Select valid Input and output features as per (regression/classification) for prediction")


            elif ML_model == 'Classification':
                if st.button("Analyze"):
                    space()
                    try:
                        if ML_algo_classification == 'logistic regression':
                            logistic_regression(x,y)
                        elif ML_algo_classification == 'decision tree':
                            decision_tree(x,y)
                        elif ML_algo_classification == 'Random Forest':
                            random_forest(x,y)
                        elif ML_algo_classification == 'KNN':
                            KNN(x,y)
                        elif ML_algo_classification == 'SVC':
                            SVC(x,y)
                    except Exception as e:
                        st.warning(" Please Select valid Input and output features as per (regression/classification) for prediction")
        except Exception as e:
            print("Error", e)
        



    st.markdown("<h4 style='text-align: center; color: #3f3f44'> @Author: Priyam Trivedi/Sravanthi Shoroff</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #3f3f44'> @2022</h4>",unsafe_allow_html=True)


if __name__ == "__main__":
    app()
