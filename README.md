# toy_RAG-enabled_slack-bot
Setting up a RAG-enabled Slack Bot on personal documents 
For a details on implementing RAGs with Cohere and Qdrant, visit: [toy_RAG](https://github.com/kvaddad1/toy_RAG)

# Setting up a RAG-Enabled Slack Bot

## Prerequisites
- Python 3.9 or higher
- Virtual environment setup
- Cohere API key
- Qdrant cloud instance
- Slack workspace admin access

## Step 1: Create a Slack App
1. Go to api.slack.com/apps
2. Click "Create New App"
   - Choose "From scratch"
   - Name it: "RAG_assistant_toydata"
   - Select your workspace

## Step 2: Configure Slack App Settings
1. Under "Socket Mode":
   - Enable Socket Mode
   - Create an App-Level Token (copy & save as SLACK_APP_TOKEN)
   - Give it the `connections:write` scope

2. Under "OAuth & Permissions":
   - Add these 'Bot Token Scopes':
     - `app_mentions:read`
     - `chat:write`
     - `channels:history`
     - `im:history`
     - `im:write`
   - Install app to workspace (check for "Install to <selected_workspace_name>" option on page)
   - Save 'Bot User OAuth Token' (as SLACK_BOT_TOKEN)

3. Under "Event Subscriptions":
   - Enable Events
   - Subscribe to bot events:
     - `message.im`
     - `message.channels`
     - `app_mention`

4. Under "App Home":
   - Enable Messages Tab
   - Enable "Allow users to send Slash commands and messages"

## Step 3: Set Up Local Environment
1. Create and activate virtual environment:
```bash
python -m venv bot_venv
source bot_venv/bin/activate  # On Windows: bot_venv\Scripts\activate
```

2. Install required packages:
```bash
pip install slack-bolt python-dotenv llama-index llama-index-core llama-index-vector-stores-qdrant llama-index-embeddings-cohere llama-index-llms-cohere cohere qdrant-client

```

3. Create `.env` file with credentials:
```env
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
COHERE_API_KEY=your-cohere-key
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key
```

## Step 4: Implement the Bot
check out simple code in `RAG_slack_bot.py` in this repo.


## Step 5: Start the Bot
1. Run the bot:
```
python RAG_slack_bot.py /path/to/your/docs
```
for demo purposes, you can use the `./papers/` folder in this repo.
```
python RAG_slack_bot.py ./papers/
```

2. Verify you see "⚡ RAG-enabled Slack bot is running!"

## Step 6: Use the Bot in Slack
1. Add bot to a channel of your choice:
   - Type `/invite @RAG_assistant_toydata` in the channel

2. Test the bot:
   - Mention the bot: `@RAG_assistant_toydata hello`
   - Ask questions about your documents: `@RAG_assistant_toydata what does the document say about...?`

## Troubleshooting
1. If messages are not working:
   - Verify all OAuth scopes are added successfully
   - Reinstall app to workspace (available on OAuth & Permissions page)
   - Check Messages Tab is enabled in App Home

2. If bot doesn't respond:
   - Check if bot is running in terminal
   - Verify documents are loaded correctly
   - Check error messages in terminal logs

## Best Practices
1. Keep the bot running using a process manager in production
2. Monitor terminal logs for errors
3. Update documents as needed by modifying DOCS_PATH
4. Use clear, specific questions for best results

## Additional Resources
- Slack API Documentation: api.slack.com/docs
- Cohere Documentation: docs.cohere.com
- Qdrant Documentation: qdrant.tech/documentation

## Use Case Example
This repository includes a practical demonstration using an impute-first workflow research paper in the `./papers` folder. The bot was successfully integrated into the Slack channel #general with the handle `@RAG_assistant_toydata`. Users can query the slack-bot about the paper's content, such as "can you please summarize impute-first work?", and receive detailed responses about the workflow's components, performance characteristics, and comparisons with other methods.

The bot successfully processed and responded with accurate information about:
- The two main components: personalization and downstream processing
- Technical details about the pangenome genotyper and imputation tool
- Comparative advantages over the Giraffe pangenome workflow

## Demo Screenshot
![RAG Slack Bot Demo](./images/rag_slack_demo.png)
