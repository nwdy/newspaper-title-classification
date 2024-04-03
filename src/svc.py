import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import re
from sklearn.metrics import accuracy_score

stopwords_vi = set([
            "và", "của", "các", "là", "cho", "trong", "với", "cũng", "này"
        ])

def preprocess_text_vi(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_vi]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df['title'].apply(preprocess_text_vi)
    y = df['genre']
    return X, y

class SVC_Classifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svm', SVC(kernel=kernel, C=C, gamma=gamma))
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def tune_parameters(self, X_train, y_train, param_grid, cv=5):
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_, grid_search.best_score_

def main():
    X_train, y_train = load_data('data/train_dataset.csv')
    X_test, y_test = load_data('data/test_dataset.csv')

    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto'],
        'svm__kernel': ['rbf', 'linear']
    }

    svc = SVC_Classifier()

    svc.train(X_train, y_train)

    pred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("Accuracy: ", accuracy)

    # Print the prediction (eg.)
    print(pred[95:105])

    best_params, best_score = svc.tune_parameters(X_train, y_train, param_grid)
    print("Best parameters:", best_params)
    print("Best cross-validation score:", best_score)


if __name__ == '__main__':
    main()