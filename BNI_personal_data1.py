import streamlit as st
import pandas as pd
from supabase import create_client, Client
from datetime import datetime, timedelta, date
import time
import logging
from typing import Optional, Dict, Any, Tuple
import traceback
from datetime import date
import calendar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules that might contain Streamlit commands AFTER set_page_config
from BNI_Personal_Data_extraction1 import extract_data_from_image

# Supabase configuration
SUPABASE_URL = "https://dvzpeyupbyqkehksvmpc.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR2enBleXVwYnlxa2Voa3N2bXBjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU3NzQwNzMsImV4cCI6MjA3MTM1MDA3M30.cKw87wSBjpqBMp42cFh5oOqRLfwOpzYysEasJ2T8llc"


class AppError(Exception):
    """Base exception for application errors"""
    pass

class DatabaseError(AppError):
    """Database related errors"""
    pass

class ValidationError(AppError):
    """Input validation errors"""
    pass

class AuthenticationError(AppError):
    """Authentication related errors"""
    pass

# Enhanced error tracking
def increment_error_count():
    """Track errors for monitoring"""
    if 'personal_data_error_count' not in st.session_state:
        st.session_state.personal_data_error_count = 0
    st.session_state.personal_data_error_count += 1
    st.session_state.personal_data_last_error = datetime.now()

def init_personal_data_session_state():
    """Initialize session state variables with unique names for personal data module"""
    if 'personal_data_logged_in' not in st.session_state:
        st.session_state.personal_data_logged_in = False
    if 'personal_data_error_count' not in st.session_state:
        st.session_state.personal_data_error_count = 0
    # UNIQUE SESSION STATE VARIABLES FOR PERSONAL DATA MODULE
    if 'personal_data_show_add_form' not in st.session_state:
        st.session_state.personal_data_show_add_form = False
    if 'personal_data_edit_member' not in st.session_state:
        st.session_state.personal_data_edit_member = None
    if 'personal_data_show_view_all_data' not in st.session_state:
        st.session_state.personal_data_show_view_all_data = False
    if 'personal_data_data_processed' not in st.session_state:
        st.session_state.personal_data_data_processed = False

def show_error_details(error_message):
    """Display error details in a user-friendly way"""
    with st.expander("🔍 Error Details"):
        st.code(error_message)

def upload_and_process_image(uploaded_file):
    """Upload image and process it to extract table data."""
    # Save uploaded file to a temporary location
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract data using the function from data_extraction.py
    extracted_data = extract_data_from_image("temp_image.png")

    return extracted_data

def handle_error(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AuthenticationError as e:
            st.error(f"Authentication failed: {str(e)}")
            logger.error(f"Authentication error in {func.__name__}: {e}")
            return None
        except ValidationError as e:
            st.error(f"Invalid input: {str(e)}")
            logger.error(f"Validation error in {func.__name__}: {e}")
            return None
        except DatabaseError as e:
            st.error(f"Database error: {str(e)}")
            logger.error(f"Database error in {func.__name__}: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return None
    return wrapper

def validate_input(field_name: str, value: Any, required: bool = False, min_length: int = 0) -> bool:
    """Validate input fields"""
    if required and (value is None or str(value).strip() == ""):
        raise ValidationError(f"{field_name} is required")

    if isinstance(value, str) and len(value.strip()) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters")

    return True

# Initialize Supabase client with error handling
@st.cache_resource
def init_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection
        client.table('bni_member_personal_details').select('id').limit(1).execute()
        return client
    except Exception as e:
        logger.error(f"Supabase initialization failed: {e}")
        raise DatabaseError(f"Failed to connect to database: {str(e)}")

try:
    supabase: Client = init_supabase()
except DatabaseError:
    st.error("❌ Database Connection Failed")
    st.info("Please check your Supabase credentials and network connection.")
    st.code("""
    Troubleshooting steps:
    1. Verify Supabase URL and API key
    2. Check network connectivity
    3. Ensure database tables exist
    4. Check Supabase service status
    """)
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }

    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .delete-confirmation {
        background-color: #fee;
        border: 1px solid #fcc;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .success-message {
        background-color: #efe;
        border: 1px solid #cfc;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .error-container {
        background-color: #fee;
        border-left: 4px solid #f56565;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def save_data_to_supabase(dataframe: pd.DataFrame):
    """Save the edited dataframe back to Supabase."""
    success_count = 0
    error_count = 0
    errors = []

    try:
        for _, row in dataframe.iterrows():
            # Clean and convert data types
            def safe_convert_to_string(value):
                """Convert value to string, return empty string if None/NaN"""
                if pd.isna(value) or value is None:
                    return None
                return str(value)

            def safe_convert_to_date(value, date_format=None):
                """
                Convert a value to a datetime object.

                - If `date_format` is provided, it will use that format to parse the date.
                - If the value is empty, None, or not a valid date, return None.
                - Handles strings, pandas Timestamps, and datetime objects.
                """

                # If it's already a datetime object or pandas Timestamp, return as datetime
                if isinstance(value, (datetime, pd.Timestamp)):
                    return value.to_pydatetime()

                try:
                    if date_format:
                        return datetime.strptime(str(value), date_format)
                    else:
                        return pd.to_datetime(value, errors='coerce').to_pydatetime()
                except (ValueError, TypeError):
                    return None

            member_data = {
                "name": safe_convert_to_string(row["name"]),
                "dob": safe_convert_to_date(row["dob"]),
                "joining_date":safe_convert_to_date(row["joining_date"]),
                "category": safe_convert_to_string(row["category"]),
                "created_at": datetime.now().isoformat(),
            }

            # Check if record already exists
            existing = supabase.table("bni_member_personal_details").select("*").eq("name", member_data["name"]).execute()

            if existing.data:
                # Update existing record
                response = supabase.table("bni_member_personal_details").update(member_data).eq("name",
                                                                                          member_data["name"]).execute()
            else:
                # Insert new record
                response = supabase.table("bni_member_personal_details").insert(member_data).execute()

            # Check if the response contains data (success)
            if response.data:
                success_count += 1
            else:
                error_count += 1
                errors.append(f"No data returned for {row['name']}")

        # Show summary message after all operations
        if success_count > 0:
            st.success(f"Successfully saved {success_count} member(s) to database!")

        if error_count > 0:
            st.error(f"Failed to save {error_count} member(s). Errors: {'; '.join(errors)}")

    except Exception as e:
        st.error(f"Database operation failed: {str(e)}")

@handle_error
def get_all_members() -> pd.DataFrame:
    """Fetch all members from database with error handling"""
    try:
        response = supabase.table("bni_member_personal_details").select('*').order('created_at', desc=True).execute()

        if response.data is None:
            logger.warning("No data returned from members query")
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        logger.info(f"Successfully fetched {len(df)} members")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch members: {e}")
        raise DatabaseError(f"Could not retrieve members: {str(e)}")

@handle_error
def add_member(name: str,  dob: datetime, joining_date:datetime, category: str) -> bool:
    """Add new member with comprehensive validation and error handling"""
    try:
        # Input validation
        validate_input("Name", name, required=True, min_length=2)

        # Prepare member data
        member_data = {
            'name': name.strip(),
            'dob': dob.isoformat(),
            'joining_date': joining_date.isoformat(),
            'category': category.strip(),
            'created_at': datetime.now().isoformat()
        }

        # Check for duplicate names (optional warning)
        existing_response = supabase.table("bni_member_personal_details").select('name').eq('name', name.strip()).execute()
        if existing_response.data:
            st.warning(f"⚠️ A member named '{name}' already exists. Adding anyway...")

        # Insert member
        response = supabase.table('bni_member_personal_details').insert(member_data).execute()

        if response.data:
            logger.info(f"Successfully added member: {name}")
            return True
        else:
            raise DatabaseError("Failed to insert member data")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error adding member: {e}")
        raise DatabaseError(f"Could not add member: {str(e)}")

@handle_error
def update_member(member_id: int, name : str, dob: datetime, joining_date:datetime, category: str) -> bool:
    """Update member with validation and error handling"""
    try:
        # Input validation
        validate_input("Name", name, required=True, min_length=2)

        # Check if member exists
        check_response = supabase.table('bni_member_personal_details').select('id').eq('id', member_id).execute()
        if not check_response.data:
            raise ValidationError(f"Member with ID {member_id} not found")

        # Prepare update data
        member_data = {
            'name': name.strip(),
            'dob': dob,
            'joining_date': joining_date,
            'category': category.strip(),
        }
        # Update member
        response = supabase.table('bni_member_personal_details').update(member_data).eq('id', member_id).execute()

        if response.data:
            logger.info(f"Successfully updated member ID: {member_id}")
            return True
        else:
            raise DatabaseError("Failed to update member data")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating member: {e}")
        raise DatabaseError(f"Could not update member: {str(e)}")

@handle_error
def delete_member(member_id):
    try:
        response = supabase.table("bni_member_personal_details").delete().eq("id", member_id).execute()
        return True
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return False

@handle_error
def search_members(query: str) -> pd.DataFrame:
    """Search members with error handling"""
    try:
        validate_input("Search query", query, required=True, min_length=1)

        # Sanitize search query
        clean_query = query.strip().replace("'", "''")  # Basic SQL injection prevention

        response = supabase.table('bni_member_personal_details').select('*').or_(
            f'name.ilike.%{clean_query}%,category.ilike.%{clean_query}%'
        ).order('created_at', desc=True).execute()

        df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        logger.info(f"Search for '{query}' returned {len(df)} results")
        return df

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise DatabaseError(f"Search failed: {str(e)}")

def show_sidebar():
    """Sidebar with image upload/capture only - PERSONAL DATA VERSION"""
    try:
        st.sidebar.markdown("")  # Ensure sidebar is created

        with st.sidebar:
            # Error monitoring
            if st.session_state.get('personal_data_error_count', 0) > 0:
                st.warning(f"⚠️ {st.session_state.personal_data_error_count} error(s) occurred this session")

            st.header("📂 Image Processing")

            capture_option = st.radio(
                "Choose input method:",
                options=["Upload File", "Capture Image"],
                key="personal_data_input_method"  # UNIQUE KEY
            )

            uploaded_file = None
            captured_image = None

            if capture_option == "Upload File":
                uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"],
                                               key="personal_data_uploader")  # UNIQUE KEY
                if uploaded_file:
                    st.success("✅ File uploaded!")
                    st.session_state.personal_data_image_to_process = uploaded_file
                    st.session_state.personal_data_image_source = "upload"
            else:
                captured_image = st.camera_input("Take a picture", key="personal_data_camera")  # UNIQUE KEY
                if captured_image:
                    st.success("✅ Image captured!")
                    st.session_state.personal_data_image_to_process = captured_image
                    st.session_state.personal_data_image_source = "capture"

            if st.session_state.get('personal_data_image_to_process') is not None:
                if st.button("🔄 Process Image", use_container_width=True, type="primary",
                           key="personal_data_process_btn"):  # UNIQUE KEY
                    with st.spinner("Processing..."):
                        try:
                            extracted_data = upload_and_process_image(st.session_state.personal_data_image_to_process)
                            if isinstance(extracted_data, list) and extracted_data:
                                extracted_data = pd.DataFrame(extracted_data)
                            else:
                                st.error("❌ No data extracted from image")
                                st.session_state.personal_data_extracted_data = None
                                return

                            st.session_state.personal_data_extracted_data = extracted_data
                            st.session_state.personal_data_data_processed = True
                            st.success("✅ Data extracted!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            st.session_state.personal_data_extracted_data = None
                            st.session_state.personal_data_data_processed = False

            if st.session_state.get('personal_data_data_processed'):
                if st.button("🗑️ Clear Data", use_container_width=True, key="personal_data_clear_btn"):  # UNIQUE KEY
                    st.session_state.personal_data_image_to_process = None
                    st.session_state.personal_data_extracted_data = None
                    st.session_state.personal_data_data_processed = False
                    st.rerun()

    except Exception as e:
        st.sidebar.error(f"Sidebar error: {str(e)}")
        logger.error(f"Sidebar error: {e}")

def show_all_members_editable():
    st.subheader("📋 All Members (Editable View)")

    df_original = get_all_members()

    if df_original is None or df_original.empty:
        st.info("No data to show.")
        return

    df_display = df_original.copy()

    # Clean up date formats
    for col in ['dob', 'joining_date']:
        df_display[col] = pd.to_datetime(df_display[col], errors='coerce').dt.date

    # Editable fields only (no add or delete via editor)
    edited_df = st.data_editor(
        df_display,
        num_rows="fixed",  # disables 'Add Row'
        use_container_width=True,
        key="personal_data_editable_member_table"  # UNIQUE KEY
    )

    # Detect changes and prepare updates
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("💾 Save Changes", key="personal_data_save_changes_btn"):  # UNIQUE KEY
            with st.spinner("Saving updated data..."):
                try:
                    for idx, row in edited_df.iterrows():
                        dob_str = row['dob'].isoformat() if isinstance(row['dob'], (datetime, date)) else row['dob']
                        joining_date_str = row['joining_date'].isoformat() if isinstance(row['joining_date'], (datetime, date)) else row['joining_date']

                        update_member(
                            member_id=row['id'],
                            name=row['name'],
                            dob=dob_str,
                            joining_date=joining_date_str,
                            category=row['category']
                        )

                    st.success("✅ All updates saved successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error saving data: {str(e)}")

    with col2:
        if st.button("⬅️ Back to Dashboard", key="personal_data_back_dashboard_from_edit"):  # UNIQUE KEY
            st.session_state.personal_data_show_view_all_data = False
            st.rerun()

    # Deletion section
    st.divider()
    st.subheader("🗑️ Delete a Member")

    selected_to_delete = st.selectbox(
        "Select a member to delete:",
        options=df_original['name'].tolist(),
        index=0 if not df_original.empty else None,
        key="personal_data_delete_selectbox"  # UNIQUE KEY
    )

    if st.button("❌ Confirm Delete", key="personal_data_confirm_delete_btn"):  # UNIQUE KEY
        try:
            member_id = df_original[df_original['name'] == selected_to_delete]['id'].values[0]
            delete_member(member_id)
            st.success(f"✅ '{selected_to_delete}' deleted successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error deleting member: {str(e)}")

# Enhanced dashboard with error monitoring
def show_dashboard():
    st.title("🏢 BNI Brilliance Members Personal Data")

    # ✅ Session flags initialization - USING UNIQUE NAMES
    if "personal_data_show_add_form" not in st.session_state:
        st.session_state.personal_data_show_add_form = False

    if "personal_data_edit_member" not in st.session_state:
        st.session_state.personal_data_edit_member = None

    if "personal_data_show_view_all_data" not in st.session_state:
        st.session_state.personal_data_show_view_all_data = False

    if "personal_data_data_processed" not in st.session_state:
        st.session_state.personal_data_data_processed = False

    # ✅ Initial UI: Show two main buttons only if no form or edit view is active
    if (
            not st.session_state.personal_data_show_add_form
            and not st.session_state.personal_data_edit_member
            and not st.session_state.personal_data_show_view_all_data
    ):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("➕ Add Member", use_container_width=True, key="personal_data_add_btn"):  # UNIQUE KEY
                st.session_state.personal_data_show_add_form = True
                st.rerun()

        with col2:
            if st.button("📋 View All Members", use_container_width=True, key="personal_data_view_all_btn"):  # UNIQUE KEY
                st.session_state.personal_data_show_view_all_data = True
                st.rerun()

        return  # Exit early

    # ✅ Add Member View
    if st.session_state.personal_data_show_add_form:
        #st.header("➕ Add New Member")
        show_add_member_form1()

    # ✅ Edit Member View
    elif st.session_state.personal_data_edit_member:
        st.header("✏️ Edit Member")
        show_edit_member_form()

    # ✅ View All Members
    elif st.session_state.personal_data_show_view_all_data:
        show_all_members_editable()

def show_dashboard_with_sidebar():
    """Combined function that shows both sidebar and dashboard"""
    # Show sidebar first
    show_sidebar()

    # Then show dashboard content
    show_dashboard()

def show_add_member_form1():
    st.header("➕ Add New Member")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("⬅️ Back to Dashboard", key="personal_data_back_dashboard_from_add_form"):  # UNIQUE KEY
            st.session_state.personal_data_show_add_form = False
            st.rerun()

    with st.form("personal_data_add_member_form"):  # UNIQUE FORM KEY
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input("Full Name*", placeholder="Enter member's full name")
            category = st.text_input("Category", placeholder="Enter the category")

        with col2:
            dob = st.date_input(
                "Date of Birth",
                value=date(1980, 1, 1),
                min_value=date(1930, 1, 1),
                max_value=date(2050, 12, 31),
                help="Select the date of birth",
                key="personal_data_dob_input"  # UNIQUE KEY
            )

        with col3:
            joining_date = st.date_input(
                "Joining Date",
                value=date.today(),
                help="Select the joining date",
                key="personal_data_joining_input"  # UNIQUE KEY
            )

        submitted = st.form_submit_button("Save Member", use_container_width=True)

        if submitted:
            with st.spinner("Adding member..."):
                result = add_member(name, dob, joining_date, category)
                if result:
                    st.success(f"✅ Member '{name}' added successfully!")
                    time.sleep(1)
                    st.session_state.personal_data_show_add_form = False
                    st.rerun()

def show_edit_member_form():
    member_data = st.session_state.personal_data_edit_member
    st.header(f"✏️ Edit Member: {member_data['name']}")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("⬅️ Back to Dashboard", key="personal_data_back_to_dashboard_btn"):  # UNIQUE KEY
            st.session_state.personal_data_edit_member = None
            st.rerun()

    try:
        st.info(
            f"**Member ID:** {member_data.get('id', 'Unknown')} | **Created:** {member_data.get('created_at', 'Unknown')}")
    except Exception as e:
        st.warning("Could not display member metadata")

    with st.form("personal_data_edit_member_form"):  # UNIQUE FORM KEY
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input("Full Name*", value=member_data.get('name', ''))
            category = st.text_input("Category", value=member_data.get('category', ''))

        with col2:
            dob = st.date_input("Date of Birth", value=member_data.get('dob', ''), key="personal_data_edit_dob")

        with col3:
            joining_date = st.date_input("Joining Date", value=member_data.get('joining_date', ''),
                                       key="personal_data_edit_joining")

        submitted = st.form_submit_button("Update Member", use_container_width=True)

        if submitted:
            with st.spinner("Updating member..."):
                member_id = member_data.get('id')
                result = update_member(member_id, name, dob, joining_date, category)
                if result:
                    st.success(f"✅ Member '{name}' updated successfully!")
                    time.sleep(1)
                    st.session_state.personal_data_edit_member = None
                    st.rerun()

def main():
    init_personal_data_session_state()
    show_dashboard_with_sidebar()

if __name__ == "__main__":
    main()

# Add this for external triggering from another file
def run_as_module():
    main()