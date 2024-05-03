import os

# Get the API key from the environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    # Use the API key in your code
    print("API Key:", api_key)
else:
    print("API Key not found. Make sure you have set the GOOGLE_API_KEY environment variable.")
