import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Naive_Bayes_Classifier:
    def __init__(self):
        self.stopwords_vi = set([
            "và", "của", "các", "là", "cho", "trong", "với", "cũng", "này"
        ])
        self.le = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.nb = MultinomialNB()

    def preprocess_text_vi(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stopwords_vi]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def train(self, X_train, y_train):
        X_train = [self.preprocess_text_vi(text) for text in X_train]
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        self.nb.fit(X_train_tfidf, y_train)

    def predict(self, X_test):
        X_test = [self.preprocess_text_vi(text) for text in X_test]
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        pred = self.nb.predict(X_test_tfidf)
        return pred
    
    def fine_tuning(self, X_train, y_train, X_test, y_test, alphas):
        X_train = [self.preprocess_text_vi(text) for text in X_train]
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test = [self.preprocess_text_vi(text) for text in X_test]
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)

        best_alpha = None
        best_accuracy = -1
        best_f1 = -1

        print("Alpha fine-tuning process: \n")
        for alpha in alphas:
            print("Alpha: ", "{:.1f}".format(alpha))
            nb = MultinomialNB(alpha=alpha)
            nb.fit(X_train_tfidf, y_train)
            pred = nb.predict(X_test_tfidf)
            
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, average='weighted')
            recall = recall_score(y_test, pred, average='weighted')
            f1 = f1_score(y_test, pred, average='weighted')
            
            print("Accuracy: ", "{:.2%}".format(accuracy))
            print("Recall: ", "{:.2%}".format(recall))
            print("Precision: ", "{:.2%}".format(precision))
            print("F1-Score: ", "{:.2%}".format(f1))
            print()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha
                best_f1 = f1
            elif accuracy == best_accuracy and f1 > best_f1:
                best_alpha = alpha
                best_f1 = f1
        
        print("Best Alpha: ", "{:.1f}".format(best_alpha))
        print("Best Accuracy: ", "{:.2%}".format(best_accuracy))
        print("Best F1-Score: ", "{:.2%}".format(best_f1))

def main():
    classifier = Naive_Bayes_Classifier()

    train_data = pd.read_csv('data-mining/refactor_code/data/train_dataset.csv')
    X_train = train_data['title']
    y_train = classifier.le.fit_transform(train_data['genre'])
    
    test_data = pd.read_csv('data-mining/refactor_code/data/test_dataset.csv')
    X_test = test_data['title']
    y_test = classifier.le.transform(test_data['genre'])

    classifier.train(X_train, y_train)
    
    pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    recall = recall_score(y_test, pred, average='weighted')
    precision = precision_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)

    print("Multinomial Naive Bayes with TFIDF: \n")
    print("Accuracy: " + str("{:.2f}".format(accuracy*100)), "%")
    print("Recall: " + str("{:.2f}".format(recall*100)), "%")
    print("Precision: " + str("{:.2f}".format(precision*100)), "%")
    print("F1-Score: " + str("{:.2f}".format(f1*100)), "%")
    print("Confusion Matrix:")
    print(cm)
    print()

    alphas = np.arange(0.1, 1.1, 0.1)
    classifier.fine_tuning(X_train, y_train, X_test, y_test, alphas)

if __name__ == "__main__":
    main()