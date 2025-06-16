import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\POOJA\NLP_PROJECT_CA2\cleaned_reviews.csv")
#print(df.head())

# Check for missing values in the cleaned_review column
#print(df['cleaned_review'].isnull().sum())
#Drop rows with missing values
df = df.dropna(subset=['cleaned_review'])

# Encode sentiment labels (positive=1, neutral=0, negative=-1)
label_encoder = LabelEncoder()
df['sentiments'] = label_encoder.fit_transform(df['sentiments'])

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['cleaned_review'])

# Convert text to sequences
X = tokenizer.texts_to_sequences(df['cleaned_review'])
X = pad_sequences(X, maxlen=100)

# Get labels
y = df['sentiments']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten


# Define LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=4),
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    # Max pooling layer to reduce dimensionality
    MaxPooling1D(pool_size=4),
    # Flatten layer to convert 2D feature maps into 1D vector
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from sklearn.utils.class_weight import compute_class_weight

# Convert classes to a NumPy array
classes = np.array([0, 1, 2])
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["sentiments"]
)

class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict)


# Function to predict sentiment
def predict_sentiment(review):
    review_seq = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=100)

    prediction = model.predict(review_pad)
    sentiment = np.argmax(prediction)  # Get highest probability class
    print(prediction)
    return label_encoder.inverse_transform([sentiment])[0]

# Test with real-time review
real_review = "Good quality product"
predicted_sentiment = predict_sentiment(real_review)
print(f"Predicted Sentiment: {predicted_sentiment}")


#Transfering Data to ui using Flask
import pickle

# Save trained model
model.save("sentiment_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle)

# Save label encoder
with open("label_encoder.pkl", "wb") as handle:
    pickle.dump(label_encoder, handle)

print("Model, tokenizer, and label encoder saved successfully!")

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import spacy
from nltk.corpus import stopwords
import string

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("sentiment_model.h5")
print("Model Loaded Successfully!")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer Loaded!")

#Load label encoder
with open("label_encoder.pkl", "rb") as handle:
    label_encoder = pickle.load(handle)
print("Label Encoder Loaded!")

# Define the API route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    print("Received Data:", data)

    review = data.get("review", "").strip()

    if not review:
        return jsonify({"error": "Empty review"}), 400

    # Preprocess input
    review_seq = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=100)

    # Predict sentiment
    prediction = model.predict(review_pad)
    sentiment = np.argmax(prediction)  # Get highest probability class
    predicted_label = label_encoder.inverse_transform([sentiment])[0]

    print("Sending Response:", predicted_label)  #Debugging: Print response
    return jsonify({"sentiment": predicted_label})

@app.route("/nlp_phases", methods=["POST"])
def nlp_phases():
    data = request.json
    review = data.get("review", "").strip()

    if not review:
        return jsonify({"error": "Empty review"}), 400

    # NLP Processing
    tokens = word_tokenize(review)
    pos_tags = pos_tag(tokens)
    filtered_text = "".join([char for char in review if char not in string.punctuation])
    doc = nlp(filtered_text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    return jsonify({
        "tokenization": tokens,
        "pos_tags": pos_tags,
        "filtered_text": filtered_text,
        "lemmatized_text": lemmatized_text
    })

# Add a simple homepage route to prevent 404 errors
@app.route('/')
def home():
    return render_template("index.html")

# Run Flask app only if this file is executed directly
if __name__ == "__main__":
    app.run(debug=False)

