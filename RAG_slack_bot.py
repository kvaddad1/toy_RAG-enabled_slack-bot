from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from custom_rag_system import RAGSystem
import os
from dotenv import load_dotenv
import logging
import sys 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Slack app
app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Initialize RAG system with pre-loaded documents
rag_system = RAGSystem(collection_name="test_collection")

# Pre-load your documents
DOCS_PATH = sys.argv[1]  # Replace with your documents path
rag_system.load_documents(DOCS_PATH)

@app.event("app_mention")
def handle_mention(event, say):
    """Handle mentions in channels"""
    try:
        # Remove the bot mention from the text
        text = event["text"]
        clean_text = text.split(">", 1)[1].strip()
        
        # Query using your existing RAG system
        response = rag_system.query(clean_text)
        say(response)
    except Exception as e:
        logger.error(f"Error in handle_mention: {str(e)}")
        say(f"Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    try:
        # Start the app
        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        print("⚡ Custom RAG System initialized successfully!")
        print(f"📚 Documents loaded from: {DOCS_PATH}")
        print("⚡ Slack bot is running!")
        print("\nTo use:")
        print("Just mention the bot (@RAG_assistant_toydata) with your questions")
        handler.start()
    except Exception as e:
        logger.error(f"Error starting the bot: {str(e)}")
