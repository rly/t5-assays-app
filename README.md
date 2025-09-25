# T5 Assays Data Assistant 🧬

A Streamlit application that connects to Google Sheets and allows users to ask questions about their data using AI through OpenRouter.

## Features

- 📊 **Data Visualization**: View your Google Sheet data in an interactive table
- 💬 **AI Chat Interface**: Ask natural language questions about your data
- 🔐 **Secure API Key Input**: Enter your OpenRouter API key securely
- 🎯 **Model Selection**: Choose from various AI models (Claude Sonnet, GPT-5, etc.)
- 📈 **Data Summary**: Get automatic insights about your dataset structure

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Google Sheets Access

Make sure your `.streamlit/secrets.toml` file is properly configured with your Google Sheets service account credentials and the Google Sheet ID you want to access. Template:

```toml
# .streamlit/secrets.toml
[connections.gsheets]
spreadsheet = "https://docs.google.com/spreadsheets/d/xxxxxxx/edit#gid=0"

# From your JSON key file
type = "service_account"
project_id = "xxx"
private_key_id = "xxx"
private_key = "xxx"
client_email = "xxx"
client_id = "xxx"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "xxx"
```

### 3. Get an OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for an account
3. Generate an API key
4. Enter the key in the app's sidebar when running

### 4. Run the Application

```bash
streamlit run assays_data_app.py
```

## Usage

1. **Start the app** and it will automatically load your Google Sheet data
2. **Enter your OpenRouter API key** in the sidebar
3. **Select your preferred AI model** from the dropdown
4. **View your data** in the left column
5. **Ask questions** in the chat interface on the right

## Example Questions

- "What are the main columns in this dataset?"
- "Can you summarize the key findings from this data?"
- "What patterns do you see in the data?"
- "Are there any outliers or anomalies?"
- "What insights can you provide about this dataset?"

## Security Notes

- API keys are entered as password fields and not stored permanently
- All data processing happens locally or through secure API calls
- Google Sheets access uses service account authentication

## Troubleshooting

- If you see "Error loading Google Sheet", check your `.streamlit/secrets.toml` configuration
- If AI responses fail, verify your OpenRouter API key is valid and has sufficient credits
- For connection issues, ensure you have internet access and the required dependencies installed