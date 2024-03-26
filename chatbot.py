import nltk
import random
import string
import warnings
import json
import os
import pickle
warnings.filterwarnings('ignore')

# User Model Management
def load_or_create_user_model(user_name, dob, likes, dislikes):
    file_path = f'user_models/{user_name}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            user_model = json.load(file)
    else:
        user_model = {"name": user_name, "personal_info": dob, "likes": likes, "dislikes": dislikes}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(user_model, file)
    return user_model

def update_user_model(user_model):
    file_path = f'user_models/{user_model["name"]}.json'
    with open(file_path, 'w') as file:
        json.dump(user_model, file)

# Assuming the knowledge_base is defined as before
knowledge_base = pickle.load(open('kb.p', 'rb'))

raw = ''
for kb in knowledge_base:
    raw += kb

raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw) # converts to list of sentences
word_tokens = nltk.word_tokenize(raw) # converts to list of words

# Preprocessing
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Vectorizer and Response Generation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response, user_model):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        chatbot_response = "I am sorry! I don't understand you"
    else:
        chatbot_response = sent_tokens[idx]
    # Personalized response based on user model
    if user_model['likes']:
        chatbot_response += f" By the way, I remember you like {''.join(user_model['likes'])}."
    return chatbot_response

def main():
    print("Hello, there! What's your name?")
    user_name = input()

    print("What is your date of birth?")
    dob = input()

    print("What do you like ?")
    likes = input()

    print("What do you dislike ?")
    dislikes = input()

    print(f"Hello, {user_name}! My name is Lyanna. I will answer your queries. If you want to exit, type Bye!")

    user_model = load_or_create_user_model(user_name, dob, likes, dislikes)

    flag = True
    while(flag):
        user_response = input()
        user_response = user_response.lower()
        if(user_response != 'bye'):
            if user_response in ('thanks', 'thank you'):
                flag = False
                print("Lyanna: You're welcome!")
            else:
                if greeting(user_response) is not None:
                    print("Lyanna: " + greeting(user_response))
                else:
                    print("Lyanna: ", end='')
                    print(response(user_response, user_model))
                    sent_tokens.remove(user_response) # This line can be adjusted based on your requirements
        else:
            flag = False
            print("Lyanna: Bye! Have a great time!")

if __name__ == "__main__":
    main()
