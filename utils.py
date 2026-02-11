import os
import re
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Get API key with multiple fallback options
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables. "
        "Please set it in your .env file with: GEMINI_API_KEY=your_api_key_here"
    )

# Configure Gemini API
try:
    genai.configure(api_key=api_key)
    thinking_model = genai.GenerativeModel("gemini-2.5-flash-thinking-exp")
    lite = genai.GenerativeModel("gemini-2.5-flash-lite")
    flash_model = genai.GenerativeModel("gemini-2.5-flash")
    logging.info("API configured successfully")
except Exception as e:
    logging.error(f"Failed to configure API: {e}")
    raise

def clean_response_text(text: str) -> str:
    """
    Clean AI response by removing:
    - Asterisks (*)
    - Backticks (`)
    - Markdown headers (# ## ###)
    - Code block markers (```python, ```, etc.)
    - Extra whitespace
    """
    if not text:
        return ""
    
    # Remove code block markers
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'```', '', text)
    
    # Remove asterisks and backticks
    text = re.sub(r'[*`]', '', text)
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def get_gemini_response(prompt: str, model_type: str = "flash", max_retries: int = 3) -> str:
    """
    Get response from Gemini API with error handling and retries.
    
    Args:
        prompt: The prompt to send to the model
        model_type: Type of model to use ("thinking", "lite", "flash")
        max_retries: Maximum number of retry attempts
    
    Returns:
        Clean response text from the model
    """
    # Select model
    if model_type.lower() == "thinking":
        model = thinking_model
    elif model_type.lower() == "lite":
        model = lite
    else:
        model = flash_model
    
    # Try to get response with retries
    for attempt in range(max_retries):
        try:
            logging.info(f"Sending request to model (attempt {attempt + 1}/{max_retries})")
            
            # Generate content
            response = model.generate_content(prompt)
            
            # Check if response has text
            if not response or not response.text:
                logging.warning(f"Empty response from model on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    continue
                return "Error: Model returned empty response"
            
            # Clean and return response
            cleaned_text = clean_response_text(response.text)
            logging.info(f"Successfully received response ({len(cleaned_text)} characters)")
            return cleaned_text
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error on attempt {attempt + 1}: {error_msg}")
            
            # Check for specific errors
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                return "Error: API quota exceeded. Please try again later."
            elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                return "Error: Authentication failed. Please check your API key."
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                if attempt < max_retries - 1:
                    logging.info("Network error, retrying...")
                    continue
                return "Error: Network connection failed."
            
            # If last attempt, return error
            if attempt == max_retries - 1:
                return f"Error: {error_msg}"
    
    return "Error: Failed to get response after multiple attempts"

def validate_api_key() -> bool:
    """
    Validate that the API key is configured correctly.
    
    Returns:
        True if API key is valid, False otherwise
    """
    try:
        # Try a simple request
        test_model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = test_model.generate_content("Say 'API configured correctly'")
        
        if response and response.text:
            logging.info("API validation successful")
            return True
        else:
            logging.error("API validation failed: No response")
            return False
            
    except Exception as e:
        logging.error(f"API validation failed: {e}")
        return False

def get_model_info() -> dict:
    """
    Get information about available models.
    
    Returns:
        Dictionary with model information
    """
    return {
        "thinking": "gemini-2.5-flash-thinking-exp - Best for complex reasoning",
        "lite": "gemini-2.5-flash-lite - Fast and efficient",
        "flash": "gemini-2.5-flash - Balanced performance"
    }

# Validate API key on module import
if __name__ != "__main__":
    try:
        if not validate_api_key():
            logging.warning("API validation failed, but continuing...")
    except Exception as e:
        logging.warning(f"Could not validate API: {e}")