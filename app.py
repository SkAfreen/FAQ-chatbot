import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

# Load FAQ dataset
with open("faq.txt", "r") as file:
    data = file.readlines()

faq = {}
for line in data:
    question, answer = line.split("?")
    faq[question.strip().lower()] = answer.strip()

# Preprocess text
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# TF-IDF and chatbot logic
questions = list(faq.keys())
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, question_vectors)
    max_index = similarity.argmax()
    if similarity[0][max_index] > 0.2:  # Threshold for matching
        return faq[questions[max_index]]
    return "Sorry, I don't understand your question."

# Streamlit UI
st.title("FAQ Chatbot")
st.write("Ask me a question below!")

user_input = st.text_input("Your question:")
if st.button("Get Response"):
    response = get_response(user_input)
    st.write("ChatBot:", response)
