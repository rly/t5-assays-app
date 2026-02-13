import streamlit as st
from streamlit_gsheets import GSheetsConnection
import requests
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import pandas as pd
import urllib.parse
import time
import plotly.express as px
import numpy as np

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

# TODO this really should be cached in the file input
# TODO some requests fail, e.g., F0207-0534 returns 400 error
def get_pubchem_urls(smiles_list: list[str]) -> list[str]:
    """Get list of Pubchem URLs for a list of SMILES strings"""
    results = []

    # URL encode SMILES strings
    smiles_list = [urllib.parse.quote(smiles) for smiles in smiles_list]

    for smiles in smiles_list:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                cids = data['IdentifierList']['CID']
                assert len(cids) == 1, f"Expected one CID for SMILES {smiles}, got {len(cids)}"
                cid = cids[0]
                urls = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
                results.append(urls)
            else:
                print(f"PubChem request for {smiles} failed: {response.status_code}")

        except Exception as e:
            print(f"Error processing {smiles}: {e}")

        # Rate limiting: PubChem allows max 5 requests per second
        time.sleep(0.25)  # 4 requests per second to be safe

    return results

model_mapping = {
    "NVIDIA: Nemotron 3 Nano 30B A3B (free)": "nvidia/nemotron-3-nano-30b-a3b:free",
    "OpenAI GPT-OSS 20B (free)": "openai/gpt-oss-20b:free",
    "Gemini 2.5 Flash ($)": "google/gemini-2.5-flash",
    "GPT-5 Mini ($)": "openai/gpt-5-mini",
    "GPT-5 ($$)": "openai/gpt-5",
    "Claude Sonnet 4 ($$$)": "anthropic/claude-4-sonnet",
}

# Custom CSS for styling
st.markdown("""
<style>
    /* Set the width of the sidebar */
    section[data-testid="stSidebar"] {
        width: 400px !important;
    }

    /* Always show the collapse button */
    div[data-testid="stSidebarCollapseButton"] {
        visibility: visible;
    }

    /* Set the padding of the main content area */
    div[data-testid="stMainBlockContainer"] {
        padding: 2rem 1rem;
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
    # Data source selection
    st.subheader("📊 Data Source")
    data_source_type = st.radio(
        "Choose data source:",
        options=["VEEV MacroD PEITHO Merge", "VEEV MacroD PARG Merge", "Google Sheet"],
        help="Select merged VEEV MacroD PARG dataset or a single Google Sheet",
        label_visibility="collapsed"
    )

    if data_source_type == "VEEV MacroD PEITHO Merge":
        st.session_state.data_source_type = "veev_peitho_merge"
        st.session_state.selected_sheet_url = None
        st.session_state.selected_sheet_id = None
        # Placeholder for merge status (will be updated after data loads)
        merge_status_placeholder = st.empty()
    elif data_source_type == "VEEV MacroD PARG Merge":
        st.session_state.data_source_type = "veev_parg_merge"
        st.session_state.selected_sheet_url = None
        st.session_state.selected_sheet_id = None
        # Placeholder for merge status (will be updated after data loads)
        parg_merge_status_placeholder = st.empty()
    else:  # data_source_type == "Google Sheet":
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
                st.session_state.selected_sheet_url = selected_sheet_info['url']
                st.session_state.selected_sheet_id = selected_sheet_info['id']
                st.session_state.data_source_type = "single_sheet"
        else:
            st.error("❌ No Google Sheets found in the specified folder")
            st.session_state.selected_sheet_url = None
            st.session_state.selected_sheet_id = None
            st.session_state.data_source_type = None

    # Data filtering options (only show for VEEV PEITHO merge)
    if data_source_type == "VEEV MacroD PEITHO Merge":
        st.subheader("🔍 Data Filters")

        # Initialize filter values in session state if not present
        if "chi2_max" not in st.session_state:
            st.session_state.chi2_max = 10.0
        if "rmse_max" not in st.session_state:
            st.session_state.rmse_max = 10.0

        chi2_max = st.number_input(
            "Chi2_ndof_RU2 <",
            min_value=0.0,
            value=st.session_state.chi2_max,
            step=1.0,
            help="Filter to include only rows where Chi2_ndof_RU2 is less than this value"
        )
        rmse_max = st.number_input(
            "RMSE_RU <",
            min_value=0.0,
            value=st.session_state.rmse_max,
            step=1.0,
            help="Filter to include only rows where RMSE_RU is less than this value"
        )
        st.session_state.chi2_max = chi2_max
        st.session_state.rmse_max = rmse_max

        # Filter buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Filters", use_container_width=True):
                st.session_state.chi2_max = 1e9
                st.session_state.rmse_max = 1e9
                st.rerun()
        with col2:
            if st.button("Reset Defaults", use_container_width=True):
                st.session_state.chi2_max = 10.0
                st.session_state.rmse_max = 10.0
                st.rerun()

        # Placeholder for filter status (will be updated after data loads)
        filter_status_placeholder = st.empty()

    # AI Model settings
    st.subheader("🤖 AI Model")
    default_api_key = st.secrets.get("openrouter", {}).get("api_key", "")
    openrouter_api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=default_api_key,
        help="Enter your OpenRouter API key to enable AI chat functionality. A default key is provided for free models only."
    )
    using_default_key = openrouter_api_key == default_api_key and default_api_key

    model = st.selectbox(
        "Select [AI Model](https://openrouter.ai/models)",
        list(model_mapping.keys()),
        index=0,
        help="Choose the AI model to use for answering questions"
    )

    if using_default_key and model not in ("OpenAI GPT-OSS 20B (free)", "NVIDIA: Nemotron 3 Nano 30B A3B (free)"):
        st.error("❌ Non-free models require your own API key.")
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
    if st.session_state.data_source_type == "veev_peitho_merge":
        # Load and merge the two VEEV MacroD PARG sheets
        with st.spinner("Loading and merging VEEV MacroD PEITHO datasets..."):
            # Get available sheets to find the URLs
            available_sheets = get_google_sheets_from_folder()

            # Find the two specific sheets
            sheet1_name = "PIETHOS_AI-docking_V2_F-converted"
            sheet2_name = "VEEV_MacroD_PEITHO_SPR_03132025_04302025_05072025"

            if sheet1_name not in available_sheets or sheet2_name not in available_sheets:
                st.error(f"❌ Could not find required sheets for merge. Looking for:\n- {sheet1_name}\n- {sheet2_name}")
                st.stop()

            # Load both sheets
            df1 = conn.read(spreadsheet=available_sheets[sheet1_name]['url'], ttl=0)
            df2 = conn.read(spreadsheet=available_sheets[sheet2_name]['url'], ttl=0)

            # Remove duplicate rows from df1
            df1 = df1.drop_duplicates()

            # Merge (outer join) the dataframes on AI Binding sheet "Name" column and Fluorescence Polarization sheet "PARG Number FP"
            df = pd.merge(df1, df2, left_on='Name', right_on='IDNUMBER', how='outer', suffixes=('_AI_Bind', '_SPR'))

            # Move "Chi2_ndof_RU2" column to the second column position
            assert "Chi2_ndof_RU2" in df.columns, "Expected 'Chi2_ndof_RU2' column in merged dataframe"
            chi2_col = df.pop("Chi2_ndof_RU2")
            df.insert(1, "Chi2_ndof_RU2", chi2_col)

            # Move "RMSE_RU" column to after Chi2_ndof_RU2
            if "RMSE_RU" in df.columns:
                rmse_col = df.pop("RMSE_RU")
                df.insert(2, "RMSE_RU", rmse_col)

            # Move "VEEV - Binding Score" column and rename it to "VEEV - AI Binding Score"
            assert "VEEV - Binding Score" in df.columns, "Expected 'VEEV - Binding Score' column in merged dataframe"
            binding_score_col = df.pop("VEEV - Binding Score")
            df.insert(3, "VEEV - AI Binding Score", binding_score_col)

            # Move KA* and KD* columns to before IDNUMBER
            ka_kd_cols = [col for col in df.columns if col.startswith(('kA', 'kD', 'KA', 'KD'))]
            insert_pos = 4  # After VEEV - AI Binding Score
            for col in ka_kd_cols:
                col_data = df.pop(col)
                df.insert(insert_pos, col, col_data)
                insert_pos += 1

            # Move "IDNUMBER" column after KA/KD columns
            assert "IDNUMBER" in df.columns, "Expected 'IDNUMBER' column in merged dataframe"
            idnumber_col = df.pop("IDNUMBER")
            df.insert(insert_pos, "IDNUMBER", idnumber_col)

            # Sort by "Chi2_ndof_RU2" column (ascending order, with NaN values last)
            df = df.sort_values(by="Chi2_ndof_RU2", ascending=True, na_position='last')

            # Replace "Structure" column with Pubchem URLs
            # assert "Structure" in df.columns, "Expected 'Structure' column in merged dataframe"
            # df["Structure"] = get_pubchem_urls(df["Structure"].astype(str).tolist())
            # TODO re-enable when the requests are reliable

            # Update merge status in sidebar
            merge_status_placeholder.success(f"✅ Merged {len(df1)} + {len(df2)} rows")

            # Apply filters from sidebar
            original_len = len(df)
            chi2_max = st.session_state.get("chi2_max", 10.0)
            rmse_max = st.session_state.get("rmse_max", 10.0)
            df["Chi2_ndof_RU2"] = pd.to_numeric(df["Chi2_ndof_RU2"], errors='coerce')
            df["RMSE_RU"] = pd.to_numeric(df["RMSE_RU"], errors='coerce')
            df = df[(df["Chi2_ndof_RU2"] < chi2_max) | df["Chi2_ndof_RU2"].isna()]
            df = df[(df["RMSE_RU"] < rmse_max) | df["RMSE_RU"].isna()]
            # Update filter status in sidebar
            filter_status_placeholder.info(f"🔍 Filtered to {len(df)} rows (from {original_len})")

    elif st.session_state.data_source_type == "veev_parg_merge":
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

            # In the AI Binding sheet, rename "VEEV - Binding Score" to "VEEV - AI Binding Score"
            df1.rename(columns={"VEEV - Binding Score": "VEEV - AI Binding Score"}, inplace=True)

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

            # Sort by "FP binding (uM)" column (ascending order, with NaN values last)
            # Create a temporary numeric column for sorting (handles string values like ">100")
            df["_sort_key"] = pd.to_numeric(df["FP binding (uM)"], errors='coerce')
            df = df.sort_values(by="_sort_key", ascending=True, na_position='last')
            df = df.drop(columns=["_sort_key"])

            # Update merge status in sidebar
            parg_merge_status_placeholder.success(f"✅ Merged {len(df1)} + {len(df2)} rows")
    elif st.session_state.data_source_type == "single_sheet":
        # Load single Google Sheet
        df = conn.read(spreadsheet=st.session_state.selected_sheet_url, ttl=0)
    else:
        st.warning("Please select a data source from the sidebar first.")
        st.stop()

    # Replace "新" in column names with "·s" (applies to all data sources)
    df.columns = df.columns.str.replace("新", "·s", regex=False)

    # Write column names to file for reference
    with open("column_names.txt", "w") as f:
        for col in df.columns:
            f.write(f"{col}\n")

    # Data View section
    st.subheader("Data View")

    # Display dataframe with Name column frozen
    st.dataframe(
        df,
        height=500,
        hide_index=True,
        column_config={"Name": st.column_config.Column(pinned=True)}
    )

    # Data summary
    st.subheader("Data Summary")

    # Scatter plot based on data source type
    if st.session_state.data_source_type == "veev_peitho_merge":
        # Scatter plot for VEEV - AI Binding Score vs Chi2_ndof_RU2 (PEITHO merge)
        x_col = "VEEV - AI Binding Score"
        y_col = "KD_M"
    elif st.session_state.data_source_type == "veev_parg_merge":
        # Scatter plot for VEEV - AI Binding Score vs FP binding (uM) (PARG merge)
        x_col = "VEEV - AI Binding Score"
        y_col = "FP binding (uM)"
    else:
        x_col = None
        y_col = None

    if st.session_state.data_source_type == "veev_peitho_merge":
        # Create correlation plots for VEEV PEITHO merge
        # Filter data with good values: RMSE < 10 and Chi2 < 10
        filtered_df = df.copy()
        if "RMSE_RU" in filtered_df.columns:
            filtered_df["RMSE_RU"] = pd.to_numeric(filtered_df["RMSE_RU"], errors='coerce')
            filtered_df = filtered_df[filtered_df["RMSE_RU"] < 10]
        if "Chi2_ndof_RU2" in filtered_df.columns:
            filtered_df["Chi2_ndof_RU2"] = pd.to_numeric(filtered_df["Chi2_ndof_RU2"], errors='coerce')
            filtered_df = filtered_df[filtered_df["Chi2_ndof_RU2"] < 10]

        st.write(f"**Filtered data (RMSE < 10, Chi2 < 10):** {len(filtered_df)} rows")

        # Define the correlation plots to create
        correlation_plots = [
            ("KD_M", "VEEV - AI Binding Score"),
            ("kA[1/(M·s)]", "VEEV - AI Binding Score"),
            ("kD[1/s]", "VEEV - AI Binding Score"),
        ]

        # Create three columns for plots
        plot_cols = st.columns(3)

        for i, (y_col, x_col) in enumerate(correlation_plots):
            with plot_cols[i]:
                if x_col in filtered_df.columns and y_col in filtered_df.columns:
                    plot_df = filtered_df[[x_col, y_col]].copy()

                    # Ensure both columns are numeric
                    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
                    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
                    plot_df = plot_df.dropna()

                    if len(plot_df) > 0:
                        # Filter to positive y values only for log scale
                        plot_df_log = plot_df[plot_df[y_col] > 0].copy()

                        if len(plot_df_log) > 0:
                            # Calculate correlation
                            correlation = np.corrcoef(plot_df_log[x_col], plot_df_log[y_col])[0, 1]

                            # Create scatter plot with log y-axis
                            fig = px.scatter(
                                plot_df_log,
                                x=x_col,
                                y=y_col,
                                trendline="ols",
                                trendline_options=dict(log_y=True),
                                title=f"{y_col} vs AI Binding (r = {correlation:.3f})",
                            )
                            fig.update_layout(
                                xaxis_title=x_col,
                                yaxis_title=y_col,
                                height=300,
                                margin=dict(l=40, r=20, t=40, b=40),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No positive y-values for {y_col} plot.")
                    else:
                        st.warning(f"No valid data for {y_col} plot.")
                else:
                    st.warning(f"Column {y_col} not found.")

        # Second row of correlation plots: KD_M vs molecular properties
        correlation_plots_2 = [
            ("KD_M", "LogP"),
            ("KD_M", "Hydrogen bonds donors"),
            ("KD_M", "Hydrogen bonds acceptors"),
        ]

        # Create three columns for second row of plots
        plot_cols_2 = st.columns(3)

        for i, (y_col, x_col) in enumerate(correlation_plots_2):
            with plot_cols_2[i]:
                if x_col in filtered_df.columns and y_col in filtered_df.columns:
                    plot_df = filtered_df[[x_col, y_col]].copy()

                    # Ensure both columns are numeric
                    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
                    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
                    plot_df = plot_df.dropna()

                    if len(plot_df) > 0:
                        # Filter to positive y values only for log scale
                        plot_df_log = plot_df[plot_df[y_col] > 0].copy()

                        if len(plot_df_log) > 0:
                            # Calculate correlation
                            correlation = np.corrcoef(plot_df_log[x_col], plot_df_log[y_col])[0, 1]

                            # Create scatter plot with log y-axis
                            fig = px.scatter(
                                plot_df_log,
                                x=x_col,
                                y=y_col,
                                trendline="ols",
                                trendline_options=dict(log_y=True),
                                title=f"KD_M vs {x_col} (r = {correlation:.3f})",
                            )
                            fig.update_layout(
                                xaxis_title=x_col,
                                yaxis_title=y_col,
                                height=300,
                                margin=dict(l=40, r=20, t=40, b=40),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No positive y-values for {x_col} plot.")
                    else:
                        st.warning(f"No valid data for {x_col} plot.")
                else:
                    st.warning(f"Column {x_col} not found.")

    elif x_col and y_col and x_col in df.columns and y_col in df.columns:
        # Original scatter plot logic for other data sources
        plot_df = df.where(df["Chi2_ndof_RU2"] < 9)
        plot_df = plot_df.where(df["KD_M"] < 10e-6)
        plot_df = plot_df[[x_col, y_col]].copy()

        # Ensure both columns are numeric
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
        plot_df = plot_df.dropna()

        if len(plot_df) > 0:
            # Filter to positive y values only for log scale
            plot_df_log = plot_df[plot_df[y_col] > 0].copy()

            if len(plot_df_log) > 0:
                # Calculate correlation with log-transformed y
                y = plot_df_log[y_col]
                correlation = np.corrcoef(plot_df_log[x_col], y)[0, 1]

                # Create scatter plot with log y-axis
                fig = px.scatter(
                    plot_df_log,
                    x=x_col,
                    y=y_col,
                    trendline="ols",
                    trendline_options=dict(log_y=True),
                    title=f"{x_col} vs {y_col} (Correlation: r = {correlation:.3f})",
                )
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No positive y-values for log-scale scatter plot.")
        else:
            st.warning(f"No valid data points for {x_col} vs {y_col} scatter plot.")


    # Add link to Google Sheet(s)
    if st.session_state.data_source_type == "single_sheet":
        sheet_url = st.session_state.selected_sheet_url
        st.markdown(f"**[Open Google Sheet]({sheet_url})**")
    elif st.session_state.data_source_type == "veev_merge":
        st.markdown("**Source Sheets:**")
        st.markdown(f"- [VEEV_MacroD_PARG_AI_Bind_09082025]({available_sheets['VEEV_MacroD_PARG_AI_Bind_09082025']['url']})")
        st.markdown(f"- [VEEV_MacroD_PARG_Fluor_Pol_07292025]({available_sheets['VEEV_MacroD_PARG_Fluor_Pol_07292025']['url']})")

    st.markdown("---")

    # Maximum number of recent messages to keep before summarizing older ones
    MAX_RECENT_MESSAGES = 6  # Keep last 3 exchanges (6 messages: 3 user + 3 assistant)

    def summarize_conversation(messages_to_summarize, openrouter_api_key, model):
        """Summarize older conversation messages to reduce context size"""
        if not messages_to_summarize:
            return None

        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages_to_summarize
        ])

        summary_prompt = (
            "Summarize the following conversation concisely, preserving key information, "
            "questions asked, and conclusions reached. Keep it brief but informative:\n\n"
            f"{conversation_text}"
        )

        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model_mapping[model],
            "messages": [
                {"role": "user", "content": summary_prompt}
            ],
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception:
            pass

        return None

    def generate_ai_response(prompt, df, openrouter_api_key, model, message_history, chi2_max, rmse_max):
        """Generate AI response for a given prompt with conversation history"""

        df = df[["Name", "Chi2_ndof_RU2", "RMSE_RU", "VEEV - AI Binding Score", "KD_M", "kA[1/(M·s)]", "kD[1/s]", "Structure", "Chemical formula", "Heavy atoms", "Rotatable bonds", "Hydrogen bonds donors", "Hydrogen bonds acceptors", "Molar refractivity", "Solubility"]].copy()

        # Prepare filter info
        filter_info = []
        if chi2_max < 1e8:
            filter_info.append(f"Chi2_ndof_RU2 < {chi2_max}")
        if rmse_max < 1e8:
            filter_info.append(f"RMSE_RU < {rmse_max}")
        filter_str = ", ".join(filter_info) if filter_info else "None"

        # Prepare data context
        data_summary = f"""
Dataset Information:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column names: {', '.join(df.columns.tolist())}
- Filters applied: {filter_str}

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

        # Build messages list with conversation history
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        # Get previous messages (excluding the current prompt which is the last message)
        previous_messages = message_history[:-1]

        # If conversation is too long, summarize older messages
        if len(previous_messages) > MAX_RECENT_MESSAGES:
            older_messages = previous_messages[:-MAX_RECENT_MESSAGES]
            recent_messages = previous_messages[-MAX_RECENT_MESSAGES:]

            # Check if we already have a summary in session state
            if "conversation_summary" not in st.session_state:
                st.session_state.conversation_summary = None

            # Notify user that context is being compacted
            st.info(f"📦 Compacting conversation history ({len(older_messages)} older messages being summarized)...")

            # Summarize older messages
            summary = summarize_conversation(older_messages, openrouter_api_key, model)
            if summary:
                st.session_state.conversation_summary = summary
                # Add summary as context
                messages.append({
                    "role": "user",
                    "content": f"[Previous conversation summary: {summary}]"
                })
                messages.append({
                    "role": "assistant",
                    "content": "I understand the context from our previous discussion. How can I help you further?"
                })

            # Add recent messages
            for msg in recent_messages:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        else:
            # Add all previous messages
            for msg in previous_messages:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add the current user prompt
        messages.append({
            "role": "user",
            "content": prompt,
        })

        data = {
            "model": model_mapping[model],
            "messages": messages,
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
            # Show model and token usage
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            if total_tokens > 0:
                st.caption(f"_Model: {model} | Tokens: {prompt_tokens:,} in + {completion_tokens:,} out = {total_tokens:,} total_")
            else:
                st.caption(f"_Model: {model}_")
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # AI Chat section at the bottom
    st.subheader("AI Chat")

    # Show context size
    ai_columns = ["Name", "Chi2_ndof_RU2", "RMSE_RU", "VEEV - AI Binding Score", "KD_M", "kA[1/(M·s)]", "kD[1/s]", "Structure", "Chemical formula", "Heavy atoms", "Rotatable bonds", "Hydrogen bonds donors", "Hydrogen bonds acceptors", "Molar refractivity", "Solubility"]
    available_cols = [col for col in ai_columns if col in df.columns]
    ai_df = df[available_cols].copy()
    context_text = ai_df.to_string()
    # Rough estimate: ~4 characters per token
    estimated_tokens = len(context_text) // 4
    st.caption(f"📊 Context: {len(ai_df)} rows × {len(available_cols)} cols (~{estimated_tokens:,} tokens)")

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
                        chi2_max = st.session_state.get("chi2_max", 10.0)
                        rmse_max = st.session_state.get("rmse_max", 10.0)
                        generate_ai_response(prompt, df, openrouter_api_key, model, st.session_state.messages, chi2_max, rmse_max)
                    except Exception as e:
                        error_msg = f"An error occurred: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # Chat input (using text_input instead of chat_input to avoid sticky bottom)
        with st.form(key="chat_form", clear_on_submit=True):
            prompt = st.text_input("Ask a question about the data...", key="chat_input", label_visibility="collapsed")
            submit_button = st.form_submit_button("Send", use_container_width=True)

        if submit_button and prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        # Quick action buttons
        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            if st.button("🔬 Summarize Results", use_container_width=True):
                # Generate dynamic prompt based on data source
                if st.session_state.data_source_type == "veev_peitho_merge":
                    sheet_names = "PIETHOS_AI-docking_V2_F-converted and VEEV_MacroD_PEITHO_SPR_03132025_04302025_05072025"
                    spr_prompt = f'These are the results from {sheet_names} merged, which represent the results of an AI Binding Assay and SPR Assay for the PEITHO compound library on the VEEV MacroDomain. Summarize these results, and in particular, note contrasts between the AI assay results (VEEV - Binding Score) and SPR assay results "Chi2_ndof_RU2".'
                elif st.session_state.data_source_type == "veev_parg_merge":
                    sheet_names = "VEEV_MacroD_PARG_AI_Bind_09082025 and VEEV_MacroD_PARG_Fluor_Pol_07292025"
                    spr_prompt = f'These are the results from {sheet_names}. e.g., "VEEV_MacroD_PARG_AI_Bind_09082025" means the results of an AI Binding Assay for the PARG compound library on the VEEV MacroDomain, done on 9/8/2025. Summarize these results, and in particular, note contrasts between the assay results.'
                elif st.session_state.data_source_type == "single_sheet":
                    sheet_name = selected_sheet_name
                    spr_prompt = f'These are the results from "{sheet_name}". e.g., "VEEV_MacroD_PARG_AI_Bind_09082025" means the results of an AI Binding Assay for the PARG compound library on the VEEV MacroDomain, done on 9/8/2025. Summarize these results.'
                else:
                    spr_prompt = "Summarize these results."

                # Add the prompt to chat and trigger response
                st.session_state.messages.append({"role": "user", "content": spr_prompt})
                # TODO don't re-run the whole app here which re-processes the data
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
