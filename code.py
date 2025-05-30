import streamlit as st
import pandas as pd
import numpy as np

# --- Initialization and Helper Functions ---

def initialize_data():
    """
    Creates and initializes the main DataFrame and unique cluster list in session state.
    The DataFrame will have 'ID', 'COL_DESCRIPTION', and 'COL_CLUSTER'.
    'ID' is set as the DataFrame index and also kept as a column.
    """
    if 'main_df' not in st.session_state:
        # Generate sample data
        num_rows = 50  # Total number of rows in the main dataset
        data = {
            'ID': range(1, num_rows + 1),
            'COL_DESCRIPTION': [
                f"This is item number {i}. It serves a unique purpose for demonstration and testing. "
                f"The description can be moderately long to ensure text wrapping is visible." 
                for i in range(1, num_rows + 1)
            ],
            'COL_CLUSTER': np.random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta'], size=num_rows)
        }
        df = pd.DataFrame(data)
        # Set 'ID' as index, but also keep it as a column for easy display and access
        st.session_state.main_df = df.set_index('ID', drop=False) 
        
        # Get unique cluster values for dropdown options
        if not st.session_state.main_df.empty:
            st.session_state.unique_clusters = sorted(st.session_state.main_df['COL_CLUSTER'].unique().tolist())
        else:
            st.session_state.unique_clusters = ["Default"] # Fallback if DataFrame is empty
    
    # Ensure unique_clusters is always initialized if main_df exists but unique_clusters was missed
    elif 'main_df' in st.session_state and 'unique_clusters' not in st.session_state:
        if not st.session_state.main_df.empty:
            st.session_state.unique_clusters = sorted(st.session_state.main_df['COL_CLUSTER'].unique().tolist())
        else:
            st.session_state.unique_clusters = ["Default"]


def get_new_sample():
    """
    Samples 10 new rows (or fewer if not enough rows are available) from the main DataFrame.
    Stores the sampled DataFrame in st.session_state.current_sample_df.
    If no rows are available in the main DataFrame, an empty DataFrame is created for the sample.
    """
    if 'main_df' in st.session_state:
        df_to_sample = st.session_state.main_df
        # Determine sample size, ensuring it doesn't exceed available rows
        sample_size = min(10, len(df_to_sample)) 

        if sample_size > 0:
            st.session_state.current_sample_df = df_to_sample.sample(n=sample_size, replace=False)
        else:
            # Create an empty DataFrame with the same columns if main_df is empty
            st.session_state.current_sample_df = pd.DataFrame(columns=df_to_sample.columns)
            # Ensure index name is consistent if it was set on the original main_df
            if df_to_sample.index.name:
                 st.session_state.current_sample_df = st.session_state.current_sample_df.set_index(df_to_sample.index.name)


# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("📝 DataFrame Row Editor & Sampler")

# Initialize data if it's the first run or session state is empty
if 'main_df' not in st.session_state:
    initialize_data()
    get_new_sample() # Get the initial sample upon first load

# Fallback for unique_clusters if it's somehow missing after initial checks
if 'unique_clusters' not in st.session_state or not st.session_state.unique_clusters:
    # This attempts to re-initialize if something went wrong, e.g. main_df became empty
    initialize_data() 
    if 'main_df' in st.session_state and st.session_state.main_df.empty and not st.session_state.get('unique_clusters'):
        st.session_state.unique_clusters = ["N/A"]


# --- UI for Control Buttons (in Sidebar) ---
st.sidebar.header("Controls")
if st.sidebar.button("🔄 New Sample", key="new_sample_button", help="Fetch 10 new random rows from the dataset."):
    get_new_sample()
    st.rerun() # Rerun the script to display the new sample

save_button = st.sidebar.button("💾 Save Changes", key="save_button", help="Save changes made to COL_CLUSTER in the current sample to the main dataset.")

# --- Displaying the Sampled DataFrame with Editable Dropdowns ---
if 'current_sample_df' in st.session_state and not st.session_state.current_sample_df.empty:
    st.subheader("Sampled Records (Edit COL_CLUSTER below):")

    # Custom header for the displayed table
    header_cols = st.columns([0.15, 0.55, 0.3]) # Column ratios: ID, Description, Cluster
    header_cols[0].markdown("**ID**")
    header_cols[1].markdown("**COL_DESCRIPTION**")
    header_cols[2].markdown("**COL_CLUSTER (Editable)**")
    st.markdown("---") # Visual separator line

    # Iterate through the sampled DataFrame to display rows and their editable dropdowns
    for idx, row_data in st.session_state.current_sample_df.iterrows():
        # 'idx' here is the original ID from the main DataFrame (since 'ID' is the index)
        display_cols = st.columns([0.15, 0.55, 0.3])

        display_cols[0].markdown(f"`{str(row_data['ID'])}`") # Display the ID
        display_cols[1].markdown(row_data['COL_DESCRIPTION']) # Display the description

        # Dropdown for COL_CLUSTER
        current_cluster_for_row = row_data['COL_CLUSTER']
        # Get global unique clusters, default to current row's cluster if list is empty
        options_for_dropdown = st.session_state.get('unique_clusters', [current_cluster_for_row])
        if not options_for_dropdown: # Ensure options_for_dropdown is never empty
            options_for_dropdown = [current_cluster_for_row] if current_cluster_for_row else ["N/A"]


        # Ensure the current row's actual cluster value is part of the dropdown options
        if current_cluster_for_row not in options_for_dropdown:
            options_for_dropdown = [current_cluster_for_row] + [opt for opt in options_for_dropdown if opt != current_cluster_for_row]
        
        try:
            # Determine the default selected index for the dropdown
            default_selection_index = options_for_dropdown.index(current_cluster_for_row)
        except ValueError:
            # Fallback if the current cluster value isn't found (e.g., None, NaN issues or data inconsistency)
            st.warning(f"Could not find cluster '{current_cluster_for_row}' for ID {idx} in options. Defaulting to first option.")
            default_selection_index = 0 
            if not options_for_dropdown: # If options became empty, provide a fallback
                 options_for_dropdown = ["N/A"]

        # The selectbox widget. Streamlit manages its state using the unique 'key'.
        # The user's selection is stored in st.session_state[f"select_{idx}"]
        display_cols[2].selectbox(
            label=" ", # Label is hidden for a cleaner look as header provides context
            options=options_for_dropdown,
            index=default_selection_index,
            key=f"select_{idx}", # Unique key composed of "select_" and the original row ID
            label_visibility="collapsed"
        )
        st.markdown("---") # Visual separator between rows

else:
    st.info("No data to display. Click 'New Sample' or ensure data is properly loaded.")
    if 'main_df' in st.session_state and st.session_state.main_df.empty:
        st.warning("The main dataset is currently empty. No samples can be drawn.")

# --- Handling the Save Button Click ---
if save_button:
    if 'current_sample_df' in st.session_state and not st.session_state.current_sample_df.empty:
        changes_made_count = 0
        # Iterate through the indices of the rows currently displayed in the sample
        for idx_in_sample in st.session_state.current_sample_df.index:
            selectbox_key = f"select_{idx_in_sample}" # Construct the key for the selectbox
            
            # Check if the selectbox state exists (it should if rendered)
            if selectbox_key in st.session_state:
                selected_new_cluster = st.session_state[selectbox_key] # Get the user's selection
                
                # Ensure the row ID from the sample exists in the main DataFrame
                if idx_in_sample in st.session_state.main_df.index:
                    # Check if the cluster value has actually changed to avoid unnecessary writes
                    if st.session_state.main_df.loc[idx_in_sample, 'COL_CLUSTER'] != selected_new_cluster:
                        # Update the COL_CLUSTER in the main DataFrame
                        st.session_state.main_df.loc[idx_in_sample, 'COL_CLUSTER'] = selected_new_cluster
                        # Also update the COL_CLUSTER in the current_sample_df to reflect the save immediately in the view
                        st.session_state.current_sample_df.loc[idx_in_sample, 'COL_CLUSTER'] = selected_new_cluster
                        changes_made_count += 1
                else:
                    # This case should be rare if sampling is correct
                    st.warning(f"Row with original ID {idx_in_sample} was not found in the main dataset. Cannot save change for this row.")
            else:
                # This indicates an issue, possibly if selectboxes weren't rendered or keys mismatched
                st.error(f"Missing selection state for row ID {idx_in_sample}. Please try sampling again.")

        if changes_made_count > 0:
            st.sidebar.success(f"✅ {changes_made_count} change(s) saved successfully to the main dataset!")
            # If new cluster values could be dynamically added (not the case here with fixed options),
            # we would update st.session_state.unique_clusters here:
            # st.session_state.unique_clusters = sorted(st.session_state.main_df['COL_CLUSTER'].unique().tolist())
            st.rerun() # Rerun to ensure UI (especially selectboxes) reflects the saved state from current_sample_df
        else:
            st.sidebar.info("ℹ️ No changes were detected to save.")
    else:
        st.sidebar.warning("⚠️ No sample data is currently loaded to save changes for.")


# --- Optional: For Debugging or Verification ---
with st.expander("View Main Dataset (First 10 Rows)"):
    if 'main_df' in st.session_state and not st.session_state.main_df.empty:
        st.dataframe(st.session_state.main_df.head(10), use_container_width=True)
    elif 'main_df' in st.session_state and st.session_state.main_df.empty:
        st.write("Main dataset is initialized but currently empty.")
    else:
        st.write("Main dataset not initialized.")

# Example of how to show the full session state for debugging purposes (can be commented out)
# with st.expander("Show Full Session State (for debugging)"):
#    st.json(st.session_state.to_dict(), expanded=False)

