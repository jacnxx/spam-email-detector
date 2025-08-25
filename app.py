import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache(allow_output_mutation=True)
def load_and_train():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['spam'] = df.label.map({'ham': 0, 'spam': 1})

    X_train, _, y_train, _ = train_test_split(df['message'], df['spam'], test_size=0.2, random_state=42)
    vect = CountVectorizer()
    X_train_dtm = vect.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_dtm, y_train)
    return vect, model

def predict_spam(texts, vect, model):
    texts_dtm = vect.transform(texts)
    preds = model.predict(texts_dtm)
    return ["Spam" if p == 1 else "Not Spam" for p in preds]

def main():
    st.title("Spam Email Detector")
    st.write("Enter text messages below to check if they are Spam or Not Spam.")

    vect, model = load_and_train()

    user_input = st.text_area("Enter messages (one per line):")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.write("Please enter some messages to classify.")
        else:
            messages = user_input.strip().split('\n')
            predictions = predict_spam(messages, vect, model)
            for i, msg in enumerate(messages):
                st.write(f"Message: {msg}")
                st.write(f"Prediction: {predictions[i]}")
                st.write("---")

if __name__ == "__main__":
    main()