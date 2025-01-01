import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL context fix for nltk
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define intents with patterns and responses
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How's it going?", "What's up?"],
        "responses": ["Hey there! 👋", "Hello! How can I help? 😊", "Hi, how's your day going? 😊", "Hello, what's up? 👋"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Goodbye", "See you", "Take care", "Bye for now"],
        "responses": ["Goodbye! 😊 See you soon! 👋", "Take care, have a great day! 🌟", "Catch you later! 😎", "See you soon! 👋 Have an awesome day! 😄"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "I appreciate it", "Thanks a lot"],
        "responses": ["You're welcome! 😊", "No problem at all! 👍", "Glad to help! 😊", "Anytime! 😊"]
    },
    {
        "tag": "time",
        "patterns": ["What time is it?", "What's the current time?", "Can you tell me the time?", "What’s the clock saying?"],
        "responses": ["I can't check the time, but you can look at your phone or computer ⏰.", "Check your device for the time! ⏱️."]
    },
    {
        "tag": "date",
        "patterns": ["What’s the date today?", "What is today’s date?", "Tell me the date", "Can you tell me the date?"],
        "responses": ["I don't have a calendar, but you can easily check the date on your phone 📅.", "You can check today's date on your device 📱."]
    },
    {
        "tag": "help",
        "patterns": ["Can you help me?", "I need assistance", "I’m stuck, help me", "What should I do next?"],
        "responses": ["Of course! What do you need help with? 🤔", "I'm here to assist! What’s the issue? 🤗", "How can I help? Let me know the problem! 💬"]
    },
    {
        "tag": "motivation",
        "patterns": ["I need some motivation", "Can you inspire me?", "Give me a motivational quote", "Tell me something inspiring"],
        "responses": ["Believe in yourself, you're capable of amazing things! 💪", "The only limit is the one you set for yourself. 🚀", "Keep going, success is just around the corner! 🏅"]
    },
    {
        "tag": "jokes",
        "patterns": ["Tell me a joke", "Make me laugh", "I need a good joke", "Do you know any funny jokes?"],
        "responses": ["Why don’t skeletons fight each other? They don’t have the guts! 💀", "I told my computer I needed a break, and it froze! ❄️", "Why don’t programmers like nature? Too many bugs! 🐛"]
    },
    {
        "tag": "productivity",
        "patterns": ["How can I be more productive?", "Give me tips on productivity", "How do I get more done?", "What’s a good productivity strategy?"],
        "responses": ["Try the Pomodoro technique: work for 25 minutes, then take a 5-minute break! ⏳", "Set clear goals, break tasks into smaller steps, and prioritize what matters! 📋", "Avoid multitasking and focus on one task at a time for better results! 🎯"]
    },
    {
        "tag": "sports",
        "patterns": ["What's the latest in sports?", "Tell me about the current sports news", "Any exciting sports events?", "Give me the latest sports updates"],
        "responses": ["I can't provide live sports news, but you can check a sports app or website! ⚽", "For the latest updates, check sports apps or your favorite sports news outlet! 🏀"]
    },
    {
        "tag": "education",
        "patterns": ["Tell me about education trends", "What’s new in education?", "Give me some education tips", "How can I improve my studies?"],
        "responses": ["Stay organized, take regular breaks, and actively engage with the material! 📚", "Try spaced repetition and active recall for better retention! 🧠", "Embrace online learning and collaborate with peers for better outcomes! 💡"]
    },
    {
        "tag": "complaints",
        "patterns": ["I have a complaint", "I'm not happy with this", "This is frustrating", "I need to complain about something"],
        "responses": ["I’m sorry to hear that. Let me know what happened, and I’ll try to help! 🙁", "I apologize if something went wrong. Please share your concern, and I'll help you resolve it! 🛠️"]
    },
    {
        "tag": "undefined",
        "patterns": ["*"],
        "responses": ["Sorry, I didn’t quite get that. 😅 Could you rephrase or ask something else?", "Oops! I didn't understand that. 🤔 Could you please ask again?"]
    }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function to generate responses based on input
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    
    # Handle undefined tags
    if tag == "undefined":
        response = random.choice(["Sorry, I didn’t quite get that. 😅", "Oops! I didn't understand that. 🤔"])
    else:
        for intent in intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                break
    
    return response

# Main function to handle Streamlit UI and chatbot interaction
def main():
    # Initialize session state for the conversation if not present
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Set page layout with custom background and header
    st.set_page_config(page_title="JD's Chatbot", page_icon="🤖", layout="wide")
    st.markdown("""
        <style>
        body {
            background-color: #f7f7f7;  /* Light background color */
            font-family: 'Arial', sans-serif;
        }
        .user-message {
            background-color: #cfe9fc;  /* Light blue for user */
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            max-width: 80%;
            margin-left: auto;
            color: #2c3e50;  /* Dark text for readability */
        }
        .bot-message {
            background-color: #ffffff;  /* White background for bot */
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            max-width: 80%;
            margin-right: auto;
            color: #2c3e50;  /* Dark text for readability */
        }
        .stTextInput>div>div>input {
            color: #2c3e50; /* Dark text input for user */
            background-color: #e1f5fe;  /* Light background color for the input box */
            border-radius: 8px;  /* Rounded corners for the input box */
            padding: 10px;
            border: 1px solid #80b3ff;  /* Light blue border for input box */
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display header and instructions
    st.title("🤖 JD's Chatbot")
    st.write("Hey there! 👋 Ready to chat? Drop a message below, and I'll respond. 😊")

    # Add a stylish chat container
    chat_container = st.container()
    
    # Display the entire conversation history
    with chat_container:
        for message in st.session_state['messages']:
            if message[0] == 'user':
                st.markdown(f'<div class="user-message">{message[1]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message[1]}</div>', unsafe_allow_html=True)

    # Input field for user message
    user_input = st.text_input("You:", "", key="input", label_visibility="collapsed")

    # Respond to the input
    if user_input:
        # Show user message
        st.session_state['messages'].append(('user', user_input))
        
        # Get chatbot response
        response = chatbot(user_input)
        
        # Show bot response
        st.session_state['messages'].append(('bot', response))
        
        # Display the new message
        st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)
        
        # Stop the app when the user says goodbye
        if "goodbye" in response.lower():
            st.write("Thanks for the chat! Have an amazing day, and see you soon! 😄")
            st.stop()

if __name__ == '__main__':
    main()
