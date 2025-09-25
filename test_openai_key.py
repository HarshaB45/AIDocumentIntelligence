import os
from dotenv import load_dotenv

# Load .env so we can read the key even if we run directly
load_dotenv()

key = os.getenv("OPENAI_API_KEY")

if not key:
    print("❌ No API key found. Make sure .env contains OPENAI_API_KEY or set the environment variable.")
else:
    # show only the first and last few chars for safety
    print("✅ API key loaded successfully:")
    print(key[:6] + "..." + key[-4:])
