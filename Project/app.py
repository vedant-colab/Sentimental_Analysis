import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import streamlit as st


data = pd.read_csv(r"Tweets.csv")
data = data[['text','airline_sentiment']]
data = data[data['airline_sentiment'] !='neutral']
tweet = data.text.values
sentiment_label = data.airline_sentiment.factorize()

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
model = tf.keras.models.load_model('my_model.h5')

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    return sentiment_label[1][prediction]


def main():
    # Add a title and a brief description
    st.title("Sentiment Analysis")
    st.markdown("""
        Enter your prompt in the text area below, and click the **Analyze** button to perform sentiment analysis.
    """)

    # Add a section for user input
    st.header("User Prompt")
    user_input = st.text_area("")
    
    # Add a button to trigger sentiment analysis
    if st.button("Analyze"):
        # Perform sentiment analysis using your model
        sentiment = predict_sentiment(user_input)
        # Display the sentiment result
        st.write("Sentiment:", sentiment)
        
# Run the app
if __name__ == "__main__":
    main()
