import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB

# SAMPLE DATASET
data = {
    'text': [
        "http://free-money-now.com",
        "http://secure-bank-login.com",
        "Win cash prizes by clicking here!",
        "https://google.com",
        "claim your reward now",
        "https://github.com",
    ],
    "label": [1, 1, 1, 0, 1, 0]  # 1 = spam, 0 = safe
}

# CREATE DATAFRAME
df= pd.DataFrame(data)

# CONVERT TEXT INTO FEATURES
vectorizer = TfidfVectorizer()
x= vectorizer.fit_transform(df['text'])

# TRAINING THE MODEL
model = MultinomialNB()
model.fit(x, df['label'])

# FUNCTION TO PREDICT IF A URL IS FAKE OR NOT
def predict_link(link):
    url_vector = vectorizer.transform([link])
    prediction = model.predict(url_vector)[0]
    return "SPAM " if prediction == 1 else "Probably Safe"

# USER INPUT
print(" 🔍 Link Spam Detector (type 'exit' to quit)\n")

while True:
    user_input = input("Enter a URL to check (like: http://free-money-now.com):  ")

    if user_input.lower() == 'exit':
        print("Exiting the Link Spam Detector. Stay safe online! 👋")
        break

    result = predict_link(user_input)
    print(f"Result: {result}\n")

