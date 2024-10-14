import React, { useState } from 'react';
import './App.css'

function App() {
  const [imageUrl, setImageUrl] = useState('');
  const [userQuery, setUserQuery] = useState('');
  const [description, setDescription] = useState('');
  const [refinedQuery, setRefinedQuery] = useState('');
  const [images, setImages] = useState([]);
  const [weblinks, setWeblinks] = useState([]);

  const handleGenerateDescription = async () => {
    try {
      const response = await fetch('http://localhost:5000/generate-description', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_url: imageUrl, user_query: userQuery })
      });
      const data = await response.json();
      setDescription(data.description);
      setRefinedQuery(data.refined_query);
    } catch (error) {
      console.error('Error generating description:', error);
    }
  };

  const handleFetchImages = async () => {
    try {
      const response = await fetch('http://localhost:5000/fetch-images', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: refinedQuery })
      });
      const data = await response.json();
      setImages(data.images);
    } catch (error) {
      console.error('Error fetching images:', error);
    }
  };
  
  const handleWebSearch = async () => {
    try {
      const response = await fetch('http://localhost:5000/fetch-weblinks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: refinedQuery })  // Ensure this matches what backend expects
      });
      const data = await response.json();
      console.log("data:", data);
      setWeblinks(data.weblinks || []);
      console.log("weblinks:", data.weblinks);
    } catch (error) {
      console.error('Error fetching web links:', error);
    }
  };

  return (
    <div className="App">
      <h1>Image Query App</h1>
      <input
        type="text"
        placeholder="Enter image URL"
        value={imageUrl}
        onChange={(e) => setImageUrl(e.target.value)}
      />
      <input
        type="text"
        placeholder="Enter your query"
        value={userQuery}
        onChange={(e) => setUserQuery(e.target.value)}
      />
      <button onClick={handleGenerateDescription}>Generate Description</button>
      
      {description && <p><strong>Description:</strong> {description}</p>}
      {refinedQuery && <p><strong>Refined Query:</strong> {refinedQuery}</p>}

      <button onClick={handleFetchImages} disabled={!refinedQuery}>Fetch Images</button>
      <button onClick={handleWebSearch} disabled={!refinedQuery}>Fetch Web Links</button>

      <div className="images">
        {images.map((src, index) => (
          <img key={index} src={src} alt={`Result ${index}`} style={{ width: '100px', margin: '10px' }} />
        ))}
      </div>
      <div>
      {/* <h2>Web Links</h2> */}
      <ul>
        {weblinks.map(({ title, link }, index) => (
          <li key={index}>
            <a href={link} target="_blank" rel="noopener noreferrer">
              {title}
            </a>
          </li>
        ))}
      </ul>
    </div>
    </div>
  );
}

export default App;




