from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests
import sys


# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

import requests
from PIL import Image
from io import BytesIO
import torch

def generate_image_description(image_url):
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Check for HTTP errors
        
        # Open the image from the response content
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    # Preprocess the image and prepare it for the model
    inputs = processor(images=image, return_tensors="pt")

    # Generate image caption
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)  # Adjust as needed

    # Decode the output to get the caption
    description = processor.decode(output[0], skip_special_tokens=True)
    description = description.replace("arafed", "").strip()
    return description

def answer_question(image_path, question):
    try:
        # Open and process the image
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    # Preprocess the image and prepare it for the model with the question
    inputs = processor(images=image, text=question, return_tensors="pt")

    # Generate answer to the question
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)  # Adjust as needed

    # Decode the output to get the answer
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer

def perform_web_search(query):
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "t": "myapp"  # You can change "myapp" to your app name
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an error for bad responses
        data = response.json()
        
        # Check for the presence of "RelatedTopics"
        if "RelatedTopics" in data:
            return data["RelatedTopics"]
        elif "Abstract" in data and data["Abstract"]:
            # Fallback if there's no "RelatedTopics" but an abstract is available
            return [{"Text": data["Abstract"], "FirstURL": data.get("AbstractURL", "")}]
        else:
            return []  # Return an empty list if no results are found
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

def yay():
    image_path = sys.argv[1]

    # Generate image description
    print(image_path)
    description = generate_image_description(image_path)
    if description:
        print(f"Image Description: {description}")

        # Ask the user for a query
        user_query = sys.argv[2]
        print(user_query)
        # Merge the image description with the user query
        # merged_query = f"{description}. {user_query}"
        # print(f"Merged Query: {merged_query}")  # For debugging
        
        return description, user_query

import spacy
from difflib import SequenceMatcher
# from blip import yay

# Load spaCy model for tokenization
nlp = spacy.load('en_core_web_sm')

# Function to check similarity between two words
def are_words_similar(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio()

# Function to check if a token is a function word (e.g., pronoun, preposition, etc.)
def is_function_word(token):
    return token.pos_ in ['PRON', 'ADP', 'DET', 'CCONJ', 'AUX', 'PART']

def reduce_description(description, user_input, threshold=0.44):
    # Process both the description and user input using spaCy
    doc_desc = nlp(description)
    doc_user = nlp(user_input)
    
    reduced_tokens = []
    
    # Loop through each token in the description
    for token_desc in doc_desc:
        # Skip function words from description
        if is_function_word(token_desc):
            # print(token_desc)
            continue
        
        # Flag to check if the token should be excluded
        token_exclude = False
        
        # Compare description words with user input words
        for token_user in doc_user:
            # Skip function words in user input as well
            if is_function_word(token_user):
                continue
            
            # Check if words are similar and above the threshold
            if are_words_similar(token_desc.lemma_, token_user.lemma_) > threshold:
                print(token_desc, token_user, are_words_similar(token_desc.lemma_, token_user.lemma_))
                token_exclude = True
                break

        # Only add the token if it's not marked for exclusion
        if not token_exclude:
            reduced_tokens.append(token_desc.text)
    
    # Join the filtered tokens to form the reduced description
    reduced_description = " ".join(reduced_tokens)
    return reduced_description

description, user_input = yay()

# Reduce the description based on user input
reduced_desc = reduce_description(description, user_input)
print(f"Reduced Description: {reduced_desc}")

def merge_and_refine(reduced_desc, user_query):
    # Merge the reduced description and user query
    combined_query = f"{user_query},{reduced_desc}"
    return combined_query

merged_query = merge_and_refine(reduced_desc, user_input)
print(merged_query)

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
# from model import merged_query

# Load T5-small model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def modify_sentence(sentence):
    # Prepend task-specific instruction (T5 expects this)
    input_sentence = f"paraphrase: {sentence} </s>"

    # Encode the input sentence
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt", max_length=512, truncation=True)

    # Generate the modified sentence
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

    # Decode the generated sentence
    modified_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove repeated words
    modified_sentence = remove_repeated_words(modified_sentence)
    return modified_sentence

def remove_repeated_words(sentence):
    words = sentence.split()
    seen = set()
    unique_words = []

    for word in words:
        # If the word is not in seen, add it to unique_words
        if word not in seen:
            unique_words.append(word)
            seen.add(word)

    # Join unique words to form the final sentence
    return ' '.join(unique_words) + " only"


sente = modify_sentence(merged_query)

def main():
    # Original sentence
    sentence = merged_query  # Make sure merged_query is defined

    # Modify the sentence using T5-small
    modified_sentence = modify_sentence(sentence)
    print(f"Original Sentence: {sentence}")
    print(f"Modified Sentence: {modified_sentence}")

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from filter import sent
import time

def fetch_images_selenium(query):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Optional: run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Remove the line that sets the binary location
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(f"https://duckduckgo.com/?q={query}&t=h_&iax=images&ia=images")

    # Wait for images to load
    WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'img')))
    
    # Scroll down to load more images if necessary
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # Allow time for new images to load

    images = driver.find_elements(By.CSS_SELECTOR, 'img')
    image_urls = [img.get_attribute('src') for img in images[:10] if img.get_attribute('src')]  # Get top 10 images

    driver.quit()
    return image_urls

# if __name__ == "__main__":
#     # merged_query = yay()
#     query = sent
#     images = fetch_images_selenium(query)
#     print(images)

from serpapi import GoogleSearch
# from filter import sent
def duckduckgo_search(query):
    params = {
        "engine": "duckduckgo",  # Specify DuckDuckGo search engine
        "q": query,              # Search query
        "api_key": "94c738ef8f54650bf8c5eed8d6394103dc0661af906219776438f24e437c5c58"  # Replace with your SERPAPI API key
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()  # Get the results as a dictionary
    
    # Extract and print the search results
    for result in results.get("organic_results", []):
        title = result.get("title")
        link = result.get("link")
        print(f"Title: {title}\nLink: {link}\n")

# Example usage
# search_query = sent
# duckduckgo_search(search_query)

def loadimages():
    search_query = sente
    images = fetch_images_selenium(search_query)
    print(images)
    
def loadwebsearch():
    search_query = sente
    duckduckgo_search(search_query)


    

