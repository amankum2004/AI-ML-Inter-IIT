# Image Query & Web Search Application

This project consists of a Flask-based backend and a React-based frontend for generating image descriptions, refining user queries, fetching images, and retrieving relevant web links. The app processes an image URL and user-defined query to provide a refined search experience.

## Features

- **Backend (Flask)**:
  - Generates image descriptions using the `BlipProcessor` model from Hugging Face.
  - Reduces descriptions using `spaCy` to focus on user-specific terms.
  - Refines queries using a `T5` model.
  - Fetches images using DuckDuckGo with Selenium.
  - Performs web search using DuckDuckGo and returns titles and links.

- **Frontend (React)**:
  - Allows users to input an image URL and query.
  - Displays generated descriptions and refined queries.
  - Fetches and displays images based on the refined query.
  - Retrieves and displays web links related to the refined query.

## Prerequisites

- **Backend**:
  - Python 3.8+
  - Flask
  - Flask-CORS
  - transformers (Hugging Face)
  - torch (PyTorch)
  - spaCy
  - Selenium
  - ChromeDriverManager
  - serpapi
  - PIL (Pillow)

- **Frontend**:
  - Node.js and npm (for React)

## Installation

### Backend Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/amankum2004/AI-ML-Inter-IIT.git
    cd your-repository/Google Lens
    cd your-repository/backend
    ```

2. **Install the required packages**:




### Frontend Setup

1. **Navigate to the frontend folder**:

    ```bash
    cd ../frontend
    ```

2. **Install dependencies**:

    ```bash
    npm install
    ```

3. **Start the React app**:

    ```bash
    npm start
    ```

    The React app should now be running at `http://localhost:3000`.

## Usage

### Backend Endpoints

- **Generate Image Description**:
   
   - **Endpoint**: `POST /generate-description`
   - **Body**:
     ```json
     {
       "image_url": "URL_of_the_image",
       "user_query": "Your search query or prompt"
     }
     ```
   - **Response**:
     ```json
     {
       "description": "Generated description",
       "reduced_description": "Reduced description",
       "refined_query": "Refined query"
     }
     ```

- **Fetch Images**:
   
   - **Endpoint**: `POST /fetch-images`
   - **Body**:
     ```json
     {
       "query": "Your search query"
     }
     ```
   - **Response**:
     ```json
     {
       "images": ["URL1", "URL2", ...]
     }
     ```

- **Fetch Web Links**:
   
   - **Endpoint**: `POST /fetch-weblinks`
   - **Body**:
     ```json
     {
       "query": "Your search query"
     }
     ```
   - **Response**:
     ```json
     {
       "weblinks": [
         {"title": "Title1", "link": "Link1"},
         {"title": "Title2", "link": "Link2"},
         ...
       ]
     }
     ```

### Frontend Interaction

1. **Input the Image URL**: Enter the URL of an image.
2. **Input the User Query**: Enter a specific query or prompt related to the image.
3. **Generate Description**: Click the button to generate a description for the image.
4. **Fetch Images**: Click to retrieve related images based on the refined query.
5. **Fetch Web Links**: Click to retrieve web links that are related to the refined query.


## Troubleshooting

- **Backend Issues**:
  - If image descriptions are not generated, ensure the model weights are downloaded correctly.
  - For Selenium errors, ensure the correct ChromeDriver version is installed.
  
- **Frontend Issues**:
  - If CORS errors occur, confirm that the Flask server is set up to allow requests from `http://localhost:3000`.
  - Check console logs for any detailed error messages.



