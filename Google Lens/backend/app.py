from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
from io import BytesIO
import requests
import torch
import spacy
from difflib import SequenceMatcher
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from flask_cors import CORS
from serpapi import GoogleSearch

app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing for communication with React frontend

# Load models and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
nlp = spacy.load('en_core_web_sm')

def generate_image_description(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return f"Error: {e}"
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=50)  # Adjust as needed
    description = processor.decode(output[0], skip_special_tokens=True)
    description = description.replace("arafed", "").strip()
    return description

def are_words_similar(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio()

def is_function_word(token):
    return token.pos_ in ['PRON', 'ADP', 'DET', 'CCONJ', 'AUX', 'PART']

def reduce_description(description, user_input, threshold=0.44):
    doc_desc = nlp(description)
    doc_user = nlp(user_input)
    reduced_tokens = []
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

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def modify_sentence(sentence):
    # Prepend task-specific instruction (T5 expects this)
    input_sentence = f"paraphrase: {sentence} </s>"
    print(input_sentence)

    # Encode the input sentence
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt", max_length=512, truncation=True)

    # Generate the modified sentence
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

    # Decode the generated sentence
    modified_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove repeated words
    print("modified sent ###########",modified_sentence)
    modified_sentence = modified_sentence.replace(".", "").strip()
    modified_sentence = remove_repeated_words(modified_sentence)
    modified_sentence = modified_sentence.replace("paraphrase: ", "").strip()
    print("after_modified sent ###########",modified_sentence)
    return modified_sentence
    
def remove_repeated_words(sentence):
    words = sentence.split()
    seen = set()
    unique_words = []
    for word in words:
        # If the word is not in seen, add it to unique_words
        if word not in seen:
            if word.lower()=='arafed':
                continue
            unique_words.append(word)
            seen.add(word)
    # Join unique words to form the final sentence
    a=' '.join(unique_words)
    print("from remove_repeated :------",a)
    return a

def fetch_images_selenium(query):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(f"https://duckduckgo.com/?q={query}&t=h_&iax=images&ia=images")
    WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'img')))
    time.sleep(5)
    images = driver.find_elements(By.CSS_SELECTOR, 'img')
    image_urls = [img.get_attribute('src') for img in images[:100] if img.get_attribute('src')]
    driver.quit()
    return image_urls

from serpapi import GoogleSearch

def duckduckgo_search(query):
    params = {
        "engine": "duckduckgo",
        "q": query,
        "api_key": "94c738ef8f54650bf8c5eed8d6394103dc0661af906219776438f24e437c5c58"  # Replace with your SERPAPI API key
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    title_links = []
    
    # Extract the search results
    for result in results.get("organic_results", []):
        title = result.get("title")
        link = result.get("link")
        if title and link:
            title_links.append({"title": title, "link": link})
    
    print(f"Fetched titles and links: {title_links}")  # For debugging
    return title_links

def merge_and_refine(reduced_desc, user_query):
    # Merge the reduced description and user query
    combined_query = f"{user_query},{reduced_desc}"
    return combined_query


@app.route('/generate-description', methods=['POST'])
def generate_description_route():
    data = request.json
    image_url = data.get('image_url')
    user_query = data.get('user_query')
    
    description = generate_image_description(image_url)
    if not description:
        return jsonify({"error": "Failed to generate description"}), 400

    reduced_desc = reduce_description(description, user_query)
    # merged_query = f"{user_query}, {reduced_desc}"
    merged_query = merge_and_refine(reduced_desc, user_query)
    print("meged -----------------------------query: ",merged_query)
    refined_query = modify_sentence(merged_query)
    print("*************************",refined_query)
    
    return jsonify({
        "description": description,
        "reduced_description": reduced_desc,
        "refined_query": refined_query
    })

@app.route('/fetch-images', methods=['POST'])
def fetch_images_route():
    data = request.json
    query = data.get('query')
    try:
        images = fetch_images_selenium(query)
        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/fetch-weblinks', methods=['POST'])
def web_search_route():
    data = request.json
    query = data.get('query')
    print(f"Received query: {query}")
    try:
        title_links = duckduckgo_search(query)
        if not title_links:
            return jsonify({"weblinks": []})  # Return an empty list if no results found
        return jsonify({"weblinks": title_links})
    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



