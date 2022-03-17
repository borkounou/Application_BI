
from classifier import Classifier

if __name__=="__main__":
    model = Classifier()
    model.svm_classifier(numerical=True,ignored_pledged=True)
    print("#"*150)
    model.naive_bayes_classifier()
    print("#"*150)
    model.knn_classifier(k=10,numerical_data=True, ignored_pledged=True)
    print("#"*150)
    model.decision_tree_classifier(numerical_data=True,ignored_pledged=True)