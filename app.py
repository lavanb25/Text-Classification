from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load and preprocess the dataset
data = pd.read_csv('dataset.csv.csv', encoding='latin-1', usecols=['v1', 'v2'])  # Use only relevant columns
data.columns = ['label', 'message']  # Rename columns for better readability

# Map labels to binary values (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 2: Create Bag of Words model
vectorizer = CountVectorizer(stop_words='english')  # Remove common stopwords
X = vectorizer.fit_transform(data['message'])  # Transform messages to numerical vectors
y = data['label']

# Step 3: Train a Naive Bayes classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the message input from the form
        user_input = request.form['message']
        # Transform the input using the vectorizer
        vectorized_input = vectorizer.transform([user_input])
        # Make a prediction
        prediction = model.predict(vectorized_input)[0]
        # Map prediction back to label
        result = "Spam" if prediction == 1 else "Ham"
        return render_template('index.html', prediction=result, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
