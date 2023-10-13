import tensorflow as tf
import numpy as np
import re 
from nltk.corpus import stopwords
import spacy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress warning message

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
stop_words.remove("not")
stop_words.remove("nor")
stop_words.remove("no")
stop_words.remove("again")
add_stopwords  = set(["movie", "film", "one", "the", "scene",
                       "this", "story", "would", "really", "and", "also", ])

stop_words = stop_words.union(add_stopwords)

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def remove_html(text):
    text = re.sub(r"<[\w]+ />", " ", text)
    text = re.sub("n't", " not", text)
    return text 

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\b\w{1,1}\b", " ", text)              
    text = re.sub(r"[^a-z]", " ", text)                                            
    text = re.sub(r"[\s]+", " ", text)                             
    return text

def process_and_filter_non_entities(text):
    doc = nlp(text)
    non_entity_lemmas = [token.lemma_ for token in doc if token.ent_type_ != "PERSON"]
    non_entity_lemmas = [token for token in non_entity_lemmas 
                         if token.lower() not in stop_words]
    text = " ".join(non_entity_lemmas) 
    return text

def preprocess_text(text):
    text_removed_html = remove_html(text)
    text_removed_ents = process_and_filter_non_entities(text_removed_html)
    processed_content = clean_text(text_removed_ents)
    return processed_content

def predict(model, processed_text):
    predictions = model.predict(np.array([processed_text]))
    
    for i in range(len(predictions)):
        print("The sentiment of the review is: ")
        if predictions[i] > 0:
            print("\nPositive :)\n")
        else:
            print("\nNegative :( \n")

    return predictions


def main():
    # Load the pre-trained model
    model_path = "./models/model_NN_final.tf"
    model = load_model(model_path)

    # Get user input from the command line
    input_text = input("Please enter a review here: ")

    # Preprocess the input text
    processed_text = preprocess_text(input_text)

    # Make predictions
    predictions = predict(model, processed_text)

    predictions

if __name__ == "__main__":
    main()
