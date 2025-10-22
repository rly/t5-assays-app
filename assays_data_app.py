import streamlit as st
from streamlit_gsheets import GSheetsConnection
import requests
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import pandas as pd

# Set the page configuration to use a wide layout
st.set_page_config(layout="wide", page_title="T5 Assays Data Assistant", page_icon="🧬")

def get_google_sheets_from_folder():
    """Get list of Google Sheets from the specified Drive folder"""
    try:
        # Create credentials from secrets
        credentials_info = {
            "type": st.secrets["connections"]["gsheets"]["type"],
            "project_id": st.secrets["connections"]["gsheets"]["project_id"],
            "private_key_id": st.secrets["connections"]["gsheets"]["private_key_id"],
            "private_key": st.secrets["connections"]["gsheets"]["private_key"],
            "client_email": st.secrets["connections"]["gsheets"]["client_email"],
            "client_id": st.secrets["connections"]["gsheets"]["client_id"],
            "auth_uri": st.secrets["connections"]["gsheets"]["auth_uri"],
            "token_uri": st.secrets["connections"]["gsheets"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["connections"]["gsheets"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["connections"]["gsheets"]["client_x509_cert_url"],
        }

        credentials = Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )

        service = build('drive', 'v3', credentials=credentials)

        # Query for Google Sheets in the specified folder
        query = f"'{st.secrets["connections"]["gsheets"]["folder_id"]}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"

        results = service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, webViewLink)"
        ).execute()

        items = results.get('files', [])

        sheets = {}
        for item in items:
            sheets[item['name']] = {
                'id': item['id'],
                'url': f"https://docs.google.com/spreadsheets/d/{item['id']}/edit",
                'webViewLink': item.get('webViewLink', '')
            }

        # Sort sheets by name
        sheets = dict(sorted(sheets.items()))

        return sheets

    except Exception as e:
        st.error(f"Error accessing Google Drive: {str(e)}")
        return {}

model_mapping = {
    "OpenAI GPT-OSS 20B (free)": "openai/gpt-oss-20b:free",
    "Gemini 2.5 Flash ($)": "google/gemini-2.5-flash",
    "GPT-5 Mini ($)": "openai/gpt-5-mini",
    "GPT-5 ($$)": "openai/gpt-5",
    "Claude Sonnet 4 ($$$)": "anthropic/claude-4-sonnet",
}

# Custom CSS for column styling
st.markdown("""
<style>
    /* Set the width of the sidebar */
    section[data-testid="stSidebar"] {
        width: 550px !important;
    }

    /* Always show the collapse button */
    div[data-testid="stSidebarCollapseButton"] {
        visibility: visible;
    }

    /* Set the padding of the main content area */
    div[data-testid="stMainBlockContainer"] {
        padding: 2rem 1rem;
    }

    /* Style the divider column */
    [data-testid="column"]:nth-child(2) {
        display: flex !important;
        justify-content: center !important;
        align-items: flex-start !important;
        padding-top: 50px !important;
    }

    /* Enhance the visual separator */
    .column-divider {
        width: 3px !important;
        background: #C7C7C7 !important;
        height: 100% !important;
        min-height: 100vh !important;
        border-radius: 2px !important;
    }

    /* Improve overall spacing */
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🧬 T5 Assays Data Assistant")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")

    # Data source selection
    st.subheader("📊 Select Data Source")

    # Radio button for data source type
    data_source_type = st.radio(
        "Choose data source:",
        options=["VEEV MacroD PARG Merge", "Google Sheet"],
        help="Select merged VEEV MacroD PARG dataset or a single Google Sheet"
    )

    if data_source_type == "Google Sheet":
        with st.spinner("Loading available sheets..."):
            available_sheets = get_google_sheets_from_folder()

        if available_sheets:
            selected_sheet_name = st.selectbox(
                "Choose a sheet:",
                options=list(available_sheets.keys()),
                help="Select which Google Sheet to analyze"
            )

            if selected_sheet_name:
                selected_sheet_info = available_sheets[selected_sheet_name]
                st.success(f"✅ Selected: {selected_sheet_name}")

                # Store selected sheet info in session state
                st.session_state.selected_sheet_url = selected_sheet_info['url']
                st.session_state.selected_sheet_id = selected_sheet_info['id']
                st.session_state.data_source_type = "single_sheet"
        else:
            st.error("❌ No Google Sheets found in the specified folder")
            st.session_state.selected_sheet_url = None
            st.session_state.selected_sheet_id = None
            st.session_state.data_source_type = None
    else:
        # VEEV MacroD PARG Merge option
        st.success("✅ Selected: VEEV MacroD PARG Merge")
        st.session_state.data_source_type = "veev_merge"
        st.session_state.selected_sheet_url = None
        st.session_state.selected_sheet_id = None

    st.markdown("---")

    # OpenRouter API key input
    default_api_key = st.secrets.get("openrouter", {}).get("api_key", "")

    openrouter_api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=default_api_key,
        help="Enter your OpenRouter API key to enable AI chat functionality. A default key is provided for free models only."
    )

    # Check if using default key
    using_default_key = openrouter_api_key == default_api_key and default_api_key

    if openrouter_api_key:
        if using_default_key:
            st.info("🔓 Using default API key (free models only)")
        else:
            st.success("✅ Custom API Key provided")
    else:
        st.warning("⚠️ Please provide an OpenRouter API key to use the chat feature")

    st.markdown("---")

    # Model selection
    model = st.selectbox(
        "Select [AI Model](https://openrouter.ai/models)",
        list(model_mapping.keys()),
        index=0,
        help="Choose the AI model to use for answering questions"
    )

    # Validate model selection with default key
    if using_default_key and model not in ("OpenAI GPT-OSS 20B (free)",):
        st.error("❌ Non-free models require your own API key. Please enter your own OpenRouter API key to use paid models.")
        model_allowed = False
    else:
        model_allowed = True

# Create a connection object to the Google Sheet
try:
    # Check if a data source is selected
    if not hasattr(st.session_state, 'data_source_type') or not st.session_state.data_source_type:
        st.warning("Please select a data source from the sidebar first.")
        st.stop()

    conn = st.connection("gsheets", type=GSheetsConnection)

    # Load data based on selected data source type
    if st.session_state.data_source_type == "single_sheet":
        # Load single Google Sheet
        df = conn.read(spreadsheet=st.session_state.selected_sheet_url, ttl=0)

        # Replace "新" in column names with "·s"
        df.columns = df.columns.str.replace("新", "·s", regex=False)

    elif st.session_state.data_source_type == "veev_merge":
        # Load and merge the two VEEV MacroD PARG sheets
        with st.spinner("Loading and merging VEEV MacroD PARG datasets..."):
            # Get available sheets to find the URLs
            available_sheets = get_google_sheets_from_folder()

            # Find the two specific sheets
            sheet1_name = "VEEV_MacroD_PARG_AI_Bind_09082025"
            sheet2_name = "VEEV_MacroD_PARG_Fluor_Pol_07292025"

            if sheet1_name not in available_sheets or sheet2_name not in available_sheets:
                st.error(f"❌ Could not find required sheets for merge. Looking for:\n- {sheet1_name}\n- {sheet2_name}")
                st.stop()

            # Load both sheets
            df1 = conn.read(spreadsheet=available_sheets[sheet1_name]['url'], ttl=0)
            df2 = conn.read(spreadsheet=available_sheets[sheet2_name]['url'], ttl=0)

            # Remove duplicate rows from df1
            df1 = df1.drop_duplicates()

            # In the AI Binding sheet, change the Name values from "PARG 1" to "PARG001"
            assert 'Name' in df1.columns, "Expected 'Name' column in AI Binding sheet"
            df1['Name'] = df1['Name'].str.replace(r'PARG (\d+)', lambda m: f'PARG{int(m.group(1)):03d}', regex=True)

            # In the AI Binding sheet, rename "VEEV - Binding Score" to "VEEV - Binding Score AI Binding"
            df1.rename(columns={"VEEV - Binding Score": "VEEV - Binding Score AI Binding"}, inplace=True)

            # In the Fluorescence Polarization sheet, there are two columns named "PARG Number". Get the column index of the second one. TODO confirm this is correct.
            assert 'PARG Number.1' in df2.columns, "Expected two 'PARG Number' columns in Fluorescence Polarization sheet"
            # Rename the second "PARG Number" column to "PARG Number FP" to avoid confusion
            df2.rename(columns={"PARG Number.1": "PARG Number FP"}, inplace=True)

            # Merge (outer join) the dataframes on AI Binding sheet "Name" column and Fluorescence Polarization sheet "PARG Number FP"
            df = pd.merge(df1, df2, left_on='Name', right_on='PARG Number FP', how='outer', suffixes=('_AI_Bind', '_FP'))

            # Move "FP binding (uM)" column to the second column position
            assert "FP binding (uM)" in df.columns, "Expected 'FP binding (uM)' column in merged dataframe"
            fp_binding_col = df.pop("FP binding (uM)")
            df.insert(1, "FP binding (uM)", fp_binding_col)

            # Move "PARG Number FP" and "PARG Number" columns to fourth and fifth positions
            parg_number_fp_col = df.pop("PARG Number FP")
            parg_number_col = df.pop("PARG Number")
            df.insert(3, "PARG Number FP", parg_number_fp_col)
            df.insert(4, "PARG Number", parg_number_col)

            st.success(f"✅ Merged {len(df1)} rows from {sheet1_name} with {len(df2)} rows from {sheet2_name}")
    else:
        st.warning("Please select a data source from the sidebar first.")
        st.stop()

    # Main layout with columns
    col1, divider, col2 = st.columns([5, 0.2, 5])

    with col1:
        st.header("Data View")

        # Display dataframe
        st.dataframe(df, height=500)

        # Data summary
        st.subheader("Data Summary")
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Columns:** {len(df.columns)}")
        st.write(f"**Column Names:** {', '.join(df.columns.tolist())}")

        # Add link to Google Sheet(s)
        if st.session_state.data_source_type == "single_sheet":
            sheet_url = st.session_state.selected_sheet_url
            st.markdown(f"**[Open Google Sheet]({sheet_url})**")
        elif st.session_state.data_source_type == "veev_merge":
            st.markdown("**Source Sheets:**")
            st.markdown(f"- [VEEV_MacroD_PARG_AI_Bind_09082025]({available_sheets['VEEV_MacroD_PARG_AI_Bind_09082025']['url']})")
            st.markdown(f"- [VEEV_MacroD_PARG_Fluor_Pol_07292025]({available_sheets['VEEV_MacroD_PARG_Fluor_Pol_07292025']['url']})")


    with divider:
        st.markdown('<div class="column-divider"></div>', unsafe_allow_html=True)

    def generate_ai_response(prompt, df, openrouter_api_key, model):
        """Generate AI response for a given prompt"""
        # Prepare data context
        data_summary = f"""
Dataset Information:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column names: {', '.join(df.columns.tolist())}

All data:
{df.to_string()}

Data types:
{df.dtypes.to_string()}
"""

        system_prompt = (
            "You are a helpful data analyst assistant. You have access to a dataset with "
            f"the following information:\n\n{data_summary}\n\nAnswer questions about this data "
            "accurately and provide insights when relevant."
        )

        # Make API call to OpenRouter
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model_mapping[model],
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]
            st.markdown(assistant_response)
            st.caption(f"_Model: {model}_")
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    with col2:
        st.header("AI Chat")

        # Chat interface
        if openrouter_api_key and model_allowed:
            # Initialize chat history in session state
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Check if the last message is from user and needs a response
            needs_response = (
                len(st.session_state.messages) > 0 and
                st.session_state.messages[-1]["role"] == "user"
            )

            # If there's a pending user message, process it
            if needs_response:
                prompt = st.session_state.messages[-1]["content"]
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            generate_ai_response(prompt, df, openrouter_api_key, model)
                        except Exception as e:
                            error_msg = f"An error occurred: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

            # Chat input
            if prompt := st.chat_input("Ask a question about the data..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            generate_ai_response(prompt, df, openrouter_api_key, model)
                        except Exception as e:
                            error_msg = f"An error occurred: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

            # Quick action buttons
            col_btn1, col_btn2 = st.columns([1, 1])

            with col_btn1:
                if st.button("🔬 Summarize Results", use_container_width=True):
                    # Generate dynamic prompt based on data source
                    if st.session_state.data_source_type == "single_sheet":
                        sheet_name = selected_sheet_name
                        spr_prompt = f'These are the results from "{sheet_name}". e.g., "VEEV_MacroD_PARG_AI_Bind_09082025" means the results of an AI Binding Assay for the PARG compound library on the VEEV MacroDomain, done on 9/8/2025. Summarize these results.'
                    elif st.session_state.data_source_type == "veev_merge":
                        sheet_names = "VEEV_MacroD_PARG_AI_Bind_09082025 and VEEV_MacroD_PARG_Fluor_Pol_07292025"
                        spr_prompt = f'These are the results from {sheet_names}. e.g., "VEEV_MacroD_PARG_AI_Bind_09082025" means the results of an AI Binding Assay for the PARG compound library on the VEEV MacroDomain, done on 9/8/2025. Summarize these results, and in particular, note contrasts between the assay results.'
                    else:
                        spr_prompt = "Summarize these results."
                    # Add the prompt to chat and trigger response
                    st.session_state.messages.append({"role": "user", "content": spr_prompt})
                    st.rerun()

            with col_btn2:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

        else:
            if not openrouter_api_key:
                st.info("👆 Please provide an OpenRouter API key in the sidebar to start asking questions about your data.")
            elif not model_allowed:
                st.info("⚠️ Please select a free model or provide your own API key to use paid models.")

            example_questions = """
#### Examples:
- What are the main columns in this dataset?
- Can you summarize the key findings from this data?
- What patterns do you see in the data?
- Are there any outliers or anomalies?
- What insights can you provide about this dataset?
"""
            st.info(example_questions)

except Exception as e:
    st.error(f"Error loading Google Sheet: {str(e)}")
    st.info("Please check your Google Sheets configuration in the .streamlit/secrets.toml file")
