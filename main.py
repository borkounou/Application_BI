from classifier import Classifier
from model import train_model

if __name__=="__main__":
    model = Classifier()
    # model.svm_classifier(numerical=True,ignored_pledged=True)
    print("#"*150)
    model.naive_bayes_classifier()
    print("#"*150)
    model.knn_classifier(k=10,numerical_data=True,ignored_pledged=True)
    print("#"*150)
    model.decision_tree_classifier(numerical_data=True,ignored_pledged=True)
    print("#"*150)
    model.svm_classifier(numerical=True,ignored_pledged=True)
    print("#"*150)
    print("SVM with all data: categorical and numerical")
    model.svm_classifier(ignored_pledged=True)
    print("#"*150)
    print("Training the multi class neural network model")
    train_model()
    