import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Load the model and vectorizer
ps = PorterStemmer()
tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
model = pickle.load(open('./model.pkl', 'rb'))

# Define the text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit app
st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.error("Please enter a message.")
    else:
        # Preprocess the input message
        transformed_sms = transform_text(input_sms)
        # Vectorize the preprocessed message
        vector_input = tfidf.transform([transformed_sms])
        # Fit the model with appropriate training data
        # Example: model.fit(X_train, y_train)
        # Predict the label
        result = model.predict(vector_input)[0]
        # Display the prediction
        if result == 1:
            st.success("Spam")
        else:
            st.success("Not Spam")
