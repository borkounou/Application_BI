from matplotlib import units
# from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm



# local package

from data_preprocess import data, main_data_splitter

class Classifier:

    def plot_confusion_matrix(self, y_test, y_pred_test, classifier):
        print("Confusion matrix of test data")
        print(confusion_matrix(y_test, y_pred_test))
        ax = plt.subplot()
        cm = confusion_matrix(y_test, y_pred_test)
        sns.heatmap(cm, annot=True, ax = ax)

        # labels, title and ticks
        ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
        ax.set_title('Confusion matrix of '+classifier)
        plt.savefig("graphs/confusion_matrix_"+classifier)


        print("Classification report of "+classifier+ " classifier on the test data ")
        
        classification = classification_report(y_test, y_pred_test)
        print(classification)

    def svm_classifier(self, numerical=False, categorical=False,ignored_pledged=False,ignored_goal=False):
        X_train, X_valid,X_test, y_train, y_valid,y_test= main_data_splitter(data,numerical=numerical, categorical=categorical, ignored_goal=ignored_goal, ignored_pledged=ignored_pledged)
        # initialize the svm classifier
        SVM = svm.LinearSVC(C=1.0,tol=1e-4, multi_class='ovr',max_iter=5000, penalty='l2') #LinearSVC
        print("Training the svm model in process...")
        SVM.fit(X_train, y_train)
        print("Training terminated")

        y_pred_valid = SVM.predict(X_valid)

        y_pred_test =  SVM.predict(X_test)

        print('Accuracy of SVM classifier on training set: {:.2f}'
            .format(SVM.score(X_train, y_train)))

        print('Accuracy of SVM classifier on validation set: {:.2f}'
            .format(SVM.score(X_valid, y_valid)))

        print("Confusion matrix of validation data")
        print(confusion_matrix(y_valid, y_pred_valid))

        print('Accuracy of SVM classifier on test set: {:.2f}'
            .format(SVM.score(X_test, y_test)))

        self.plot_confusion_matrix(y_test, y_pred_test, "svm")
        

    def naive_bayes_classifier(self):
        X_train, X_valid,X_test, y_train, y_valid,y_test= main_data_splitter(data, bayesian=True)
        naive = CategoricalNB()
        print("Training the naive bayes model in process...")
        naive.fit(X_train, y_train)
        print("Training terminated")

        y_pred_valid = naive.predict(X_valid)

        y_pred_test =  naive.predict(X_test)


        print('Accuracy of naive bayes classifier on training set: {:.2f}'
            .format(naive.score(X_train, y_train)))

        print('Accuracy of bayes classifier on validation set: {:.2f}'
            .format(naive.score(X_valid, y_valid)))

        print("Confusion matrix of validation data")
        print(confusion_matrix(y_valid, y_pred_valid))

        print('Accuracy of naive bayes classifier on test set: {:.2f}'
            .format(naive.score(X_test, y_test)))

        
        self.plot_confusion_matrix(y_test, y_pred_test, "naive")

    def knn_classifier(self, k=20,numerical_data=False,categorical_data=False,ignored_pledged=False,ignored_goal=False):
        X_train, X_valid,X_test, y_train, y_valid,y_test= main_data_splitter(data, numerical=numerical_data,categorical=categorical_data,ignored_pledged=ignored_pledged,ignored_goal=ignored_goal)
        print("Training the k nearest neighbors model in process...")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        print("Training terminated")

        print("Training terminated")

        y_pred_valid = knn.predict(X_valid)

        y_pred_test =  knn.predict(X_test)


        print('Accuracy of  knn classifier on training set: {:.2f}'
            .format(knn.score(X_train, y_train)))

        print('Accuracy of knn classifier on validation set: {:.2f}'
            .format(knn.score(X_valid, y_valid)))

        print("Confusion matrix of validation data")
        print(confusion_matrix(y_valid, y_pred_valid))

        print('Accuracy of knn classifier on test set: {:.2f}'
            .format(knn.score(X_test, y_test)))
        
        self.plot_confusion_matrix(y_test, y_pred_test, "knn")

    def decision_tree_classifier(self,numerical_data=False,categorical_data=False,ignored_pledged=False,ignored_goal=False):
        X_train, X_valid,X_test, y_train, y_valid,y_test= main_data_splitter(data, numerical=numerical_data,categorical=categorical_data,ignored_pledged=ignored_pledged,ignored_goal=ignored_goal)
        dectree = tree.DecisionTreeClassifier()
        dectree.fit(X_train, y_train)
        print("Training terminated")
        y_pred_valid = dectree.predict(X_valid)

        y_pred_test =  dectree.predict(X_test)
        print('Accuracy of  dectree classifier on training set: {:.2f}'
            .format(dectree.score(X_train, y_train)))
        print('Accuracy of dectree classifier on validation set: {:.2f}'
            .format(dectree.score(X_valid, y_valid)))
        print("Confusion matrix of validation data")
        print(confusion_matrix(y_valid, y_pred_valid))
        print('Accuracy of dectree classifier on test set: {:.2f}'
            .format(dectree.score(X_test, y_test)))
        
        self.plot_confusion_matrix(y_test, y_pred_test, "dectree")

model = Classifier()
# model.svm_classifier(numerical=True,ignored_pledged=True)
# # print("#"*150)
model.naive_bayes_classifier()
# print("#"*150)
# model.knn_classifier(k=30,numerical_data=True, ignored_pledged=True)
# print("#"*150)
# model.decision_tree_classifier(numerical_data=True, ignored_pledged=True)