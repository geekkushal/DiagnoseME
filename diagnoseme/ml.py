import pandas as pd
import numpy as np
from sklearn import svm
from django.conf import settings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

csv_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'training.csv')

def getSymptomskush(sym,symptoms):
    X_test = []
    for i in symptoms:
        if i in sym:
            X_test.append(1)
        else:
            X_test.append(0)
    return X_test


def predict_disease_by_kushal(sym):

    total_data = pd.read_csv(csv_file_path)
    symptoms = pd.read_csv(csv_file_path, nrows=1).columns.tolist()
    symptoms.pop()  
    X = total_data.iloc[:, 0:132]  
    y = total_data.iloc[:, 132] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

  
    classifier = RandomForestClassifier(random_state=1)

    #  hyperparameters grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }


    random_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_dist,
                                       n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)

    random_search.fit(X_train, y_train)

    best_classifier = random_search.best_estimator_

    X_test_symptoms = getSymptomskush(sym, symptoms)  
    X_test_symptoms = sc.transform([X_test_symptoms])
    predicted_disease = best_classifier.predict(X_test_symptoms)
    return predicted_disease[0]

def predict_disease_by_kushal2(sym):

    total_data = pd.read_csv(csv_file_path)
    symptoms = pd.read_csv(csv_file_path, nrows=1).columns.tolist()
    symptoms.pop()  


    X = total_data.iloc[:, 0:132]  
    y = total_data.iloc[:, 132]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

     # Models
    models = {
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42),
    }

    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)


        X_test_symptoms = getSymptomskush(sym, symptoms) 
        X_test_symptoms = sc.transform([X_test_symptoms]) 
        predicted_disease = model.predict(X_test_symptoms)

        predictions[model_name] = predicted_disease[0]

    #print(predictions)
    return predictions

def plot_model_performance(model_metrics):

    accuracies = {model: metrics['accuracy'] for model, metrics in model_metrics.items()}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
    plt.title("Model Accuracies")
    plt.ylabel("Accuracy Score")
    plt.show()

    for model, metrics in model_metrics.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt="d", cmap="coolwarm")
        plt.title(f"Confusion Matrix - {model}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    
    for model, metrics in model_metrics.items():
        print(f"Classification Report for {model}:\n")
        print(metrics['classification_report'])
def evaluate_models():

    total_data = pd.read_csv(csv_file_path)
    symptoms = pd.read_csv(csv_file_path, nrows=1).columns.tolist()
    symptoms.pop()  

    X = total_data.iloc[:, 0:132] 
    y = total_data.iloc[:, 132]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42),
    }

    model_metrics = {}

    for model_name, model in models.items():
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = round(accuracy_score(y_test, y_pred), 4) 
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        model_metrics[model_name] = {
            'accuracy': accuracy,  
            'confusion_matrix': cm.tolist(),  
            'classification_report': report
        }

   # print(model_metrics)

    return model_metrics



















##############################################################







# Step 1: Load and Preprocess the Dataset
def load_tongue_data(data_dir, img_size=(64, 64)):
    classes = sorted(os.listdir(data_dir))  # Dynamically get class names
    num_classes = len(classes)
    x, y = [], []

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            x.append(img / 255.0)  # Normalize to [0, 1]
            y.append(label)

    x = np.array(x)
    y = np.eye(num_classes)[np.array(y)]  # One-hot encode labels
    return x, y, num_classes

# Step 2: Visualize Random Examples
def plot_random_examples(x, y, p=None):
    indices = np.random.choice(range(0, x.shape[0]), 10)
    y = np.argmax(y, axis=1)
    if p is None:
        p = y
    plt.figure(figsize=(10, 5))
    for i, index in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[index])
        plt.xticks([])
        plt.yticks([])
        col = 'g' if y[index] == p[index] else 'r'
        plt.xlabel(str(p[index]), color=col)
    plt.show()

# Step 3: Define the Neural Network
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.L = len(layers)
        self.num_features = layers[0]
        self.num_classes = layers[-1]

        self.W = {}
        self.b = {}
        self.dW = {}
        self.db = {}

        self.setup()

    def setup(self):
        for i in range(1, self.L):
            self.W[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i - 1])))
            self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    def forward_pass(self, X):
        A = tf.convert_to_tensor(X, dtype=tf.float32)
        for i in range(1, self.L):
            Z = tf.matmul(A, tf.transpose(self.W[i])) + tf.transpose(self.b[i])
            A = tf.nn.relu(Z) if i != self.L - 1 else Z
        return A

    def compute_loss(self, A, Y):
        loss = tf.nn.softmax_cross_entropy_with_logits(Y, A)
        return tf.reduce_mean(loss)

    def update_params(self, lr):
        for i in range(1, self.L):
            self.W[i].assign_sub(lr * self.dW[i])
            self.b[i].assign_sub(lr * self.db[i])

    def train_on_batch(self, X, Y, lr):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape: # Changed to persistent=True
            A = self.forward_pass(X)
            loss = self.compute_loss(A, Y)

        for i in range(1, self.L):
            self.dW[i] = tape.gradient(loss, self.W[i])
            self.db[i] = tape.gradient(loss, self.b[i])

        del tape # Delete the tape after calculating all gradients

        self.update_params(lr)
        return loss.numpy()

    def train(self, x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr):
        history = {'val_loss': [], 'train_loss': [], 'val_acc': []}

        for e in range(epochs):
            epoch_train_loss = 0.0
            print(f'Epoch {e}', end=' ')
            for i in range(steps_per_epoch):
                x_batch = x_train[i * batch_size:(i + 1) * batch_size]
                y_batch = y_train[i * batch_size:(i + 1) * batch_size]
                batch_loss = self.train_on_batch(x_batch, y_batch, lr)
                epoch_train_loss += batch_loss
                if steps_per_epoch >= 10 and i % (steps_per_epoch // 10) == 0:
                    print('.', end='')

            history['train_loss'].append(epoch_train_loss / steps_per_epoch)

            val_A = self.forward_pass(x_test)
            val_loss = self.compute_loss(val_A, y_test).numpy()
            history['val_loss'].append(val_loss)

            val_preds = self.predict(x_test)
            val_acc = np.mean(np.argmax(y_test, axis=1) == val_preds.numpy())
            history['val_acc'].append(val_acc)

            print(f' Val acc: {val_acc:.4f}')
        return history

    def predict(self, X):
        A = self.forward_pass(X)
        return tf.argmax(tf.nn.softmax(A), axis=1)

    def info(self):
        num_params = 0
        for i in range(1, self.L):
            num_params += self.W[i].shape[0] * self.W[i].shape[1]
            num_params += self.b[i].shape[0]
        print(f'Input Features: {self.num_features}')
        print(f'Number of Classes: {self.num_classes}')
        print('Hidden Layers:')
        for i in range(1, self.L - 1):
            print(f'Layer {i}, Units {self.layers[i]}')
        print(f'Number of parameters: {num_params}')

# Step 4: Load and Train the Model
DATA_DIR = os.path.join(settings.BASE_DIR, 'static', 'data', 'tongue_dataset', 'train')
#DATA_DIR = "{% static 'data/tongue_dataset/train/' %}"
#DATA_DIR = 'E:\minor_codes\Project2-20241119T114829Z-001\Project2\data\tongue_dataset\train'
x_data, y_data, num_classes = load_tongue_data(DATA_DIR)

split_idx = int(0.8 * len(x_data))
x_train, y_train = x_data[:split_idx], y_data[:split_idx]
x_test, y_test = x_data[split_idx:], y_data[split_idx:]

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

net = NeuralNetwork([64 * 64 * 3, 128, 128, num_classes])
net.info()

batch_size = 32
epochs = 10
steps_per_epoch = len(x_train) // batch_size
lr = 3e-3

history = net.train(x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr)

# Step 5: Visualize Training Results
def plot_results(history):
    plt.figure(figsize=(12, 4))
    epochs = len(history['val_loss'])
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history['val_loss'], label='Val Loss')
    plt.plot(range(epochs), history['train_loss'], label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def test_single_image(image_path, model, img_size=(64, 64)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size) / 255.0
    img_flatten = img.reshape(1, -1)
    prediction = model.predict(img_flatten)
    return prediction.numpy()[0]

#image_path = "/content/drive/MyDrive/Project2/data/tongue_dataset/test/greasy/e54624645824a4981b6bf8be3ab09fe3f-4457-0-2_jpg.rf.72898fc845e6de914e83d30cae03a178.jpg"  # Replace with your test image
image_path = os.path.join(
    settings.BASE_DIR, 
    'static', 
    'data', 
    'tongue_dataset', 
    'test', 
    'greasy', 
    'e38a42ba4b60148379eea5fdb6c6b2165-4250-0-2_jpg.rf.2fcd448414fd78a0c6abc4004fc3e3cd.jpg'
)

#D:\minor_project\minor\static\data\tongue_dataset\test\greasy\e38a42ba4b60148379eea5fdb6c6b2165-4250-0-2_jpg.rf.2fcd448414fd78a0c6abc4004fc3e3cd.jpg


plot_results(history)

# Step 6: Evaluate the Model
val_preds = net.predict(x_test)
val_accuracy = np.mean(np.argmax(y_test, axis=1) == val_preds.numpy())
print(f'Validation Accuracy: {val_accuracy:.4f}')

def predict_using_image(image_path):
    classes = sorted(os.listdir(DATA_DIR))
    predicted_class = test_single_image(image_path, net)
    return classes[predicted_class]





