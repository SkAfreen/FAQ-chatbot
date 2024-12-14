import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the FAQ dataset
with open("faq.txt", "r") as file:
    data = file.readlines()

# Parse questions and answers
faq = {}
for line in data:
    question, answer = line.split("?")
    faq[question.strip().lower()] = answer.strip()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return words




# Prepare questions for vectorization
questions = list(faq.keys())
vectorizer = TfidfVectorizer()  # Create the vectorizer
question_vectors = vectorizer.fit_transform(questions)  # Fit and transform questions


def get_response(user_input):
    user_vec = vectorizer.transform([user_input])  # Vectorize user input
    similarity = cosine_similarity(user_vec, question_vectors)  # Compare with question_vectors

    max_index = similarity.argmax()  # Find the most similar question
    if similarity[0][max_index] > 0.2:  # Threshold for matching
        return faq[questions[max_index]]
    return "Sorry, I don't understand your question."

print("ChatBot: Hello! Ask me anything or type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("ChatBot: Goodbye!")
        break
    response = get_response(user_input)
    print("ChatBot:", response)


