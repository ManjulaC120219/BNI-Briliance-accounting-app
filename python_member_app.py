import streamlit as st

# Configure Streamlit page
'''
st.set_page_config(
    page_title="BNI Brilliance Member Management",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)
'''
import pandas as pd
import math
from supabase import create_client, Client
import hashlib
from datetime import datetime, timedelta, date
import time
import io
import logging
from typing import Optional, Dict, Any, Tuple
import calendar
import traceback
# At the top of python_member_app.py, change the import to:
import BNI_personal_data1 as personal_data_module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from Final_bni_data_extraction import extract_data_from_image_v2

def show_back_to_main_button():
    """Show back to main dashboard button"""
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("⬅️ Back to Main", key="back_to_main_member_app", use_container_width=True):
            st.session_state.selected_app = None
            st.session_state.logged_in = True
            st.rerun()


# Supabase configuration
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]


# Error handling utilities
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
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    st.session_state.error_count += 1
    st.session_state.last_error = datetime.now()


def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0


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
    extracted_data = extract_data_from_image_v2("temp_image.png")

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
        client = create_client(supabase_url, supabase_key)
        # Test connection
        client.table('image_data_extraction').select('id').limit(1).execute()
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


def save_data_to_supabase(dataframe: pd.DataFrame, selected_date=None):
    """Save extracted data to Supabase with optional date"""
    try:
        # Use the selected_date parameter if provided, otherwise use today
        if selected_date is None:
            selected_date = datetime.now().date()

        # Ensure selected_date is a date object
        if isinstance(selected_date, datetime):
            selected_date = selected_date.date()

        # Combine selected date with current time for timestamp
        selected_datetime = datetime.combine(selected_date, datetime.now().time())

        # Drop 'id' if present
        if "id" in dataframe.columns:
            dataframe = dataframe.drop(columns=["id"])

        records_to_insert = []

        for _, row in dataframe.iterrows():
            # Clean and prepare the payment value
            payment_value = str(row.get('Payment', 0))
            payment_value = payment_value.replace('Rs.', '').replace('₹', '').replace(',', '').strip()

            try:
                payment_amount = int(float(payment_value))
            except (ValueError, TypeError):
                payment_amount = 0

            record = {
                'name': str(row.get('Name', '')).strip(),
                'toa': str(row.get('TOA', '')).strip(),
                'payment': payment_amount,
                'mode': str(row.get('Mode', '')).strip(),
                'created_at': selected_datetime.isoformat()  # ✅ Use selected date
            }
            records_to_insert.append(record)

        # Insert all records
        if records_to_insert:
            response = supabase.table('image_data_extraction').insert(records_to_insert).execute()

            if response.data:
                st.success(
                    f"✅ {len(records_to_insert)} records saved successfully for {selected_date.strftime('%Y-%m-%d')}!")

                # Clear the processed data
                st.session_state.extracted_data = None
                st.session_state.data_processed = False
                st.session_state.image_to_process = None

                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to save records to database")
        else:
            st.warning("No valid records to save")

    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

def debug_supabase_connection():
    """Debug function to test Supabase connection"""
    try:
        supabase = init_supabase()
        if not supabase:
            return "❌ Supabase client is None"

        # Test basic connection
        response = supabase.table("image_data_extraction").select("*").limit(1).execute()
        return f"✅ Connection OK. Sample data: {response.data}"

    except Exception as e:
        return f"❌ Connection failed: {str(e)}"


# Alternative simplified save function for testing
def save_data_simple_test(dataframe: pd.DataFrame):
    """Simplified version for testing with correct schema"""
    try:
        supabase = init_supabase()

        # Helper functions
        def safe_int_convert(value):
            if pd.isna(value) or value == "" or value is None:
                return None
            try:
                return int(float(str(value)))
            except (ValueError, TypeError):
                return None

        # Test with just one record
        first_row = dataframe.iloc[0]
        test_data = {
            "name": str(first_row.get("Name", "")).strip(),
            "payment": safe_int_convert(first_row.get("Payment", "")),  # int8
            "toa": str(first_row.get("TOA", "")).strip(),
            "mode": str(first_row.get("Mode", "")).strip(),
            "created_at": datetime.now().isoformat()
        }

        st.write("Testing with data:", test_data)
        st.write("Payment value type:", type(test_data["payment"]))

        response = supabase.table("image_data_extraction").insert(test_data).execute()

        st.write("Response:", response.data)
        st.write("Error:", getattr(response, 'error', 'None'))

        return response.data is not None

    except Exception as e:
        st.error(f"Test failed: {str(e)}")
        st.write(traceback.format_exc())
        return False


# Database helper functions with enhanced error handling
def hash_password(password: str) -> str:
    """Hash password using SHA-256 with input validation"""
    try:
        validate_input("Password", password, required=True, min_length=1)
        return hashlib.sha256(password.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        raise ValidationError("Failed to process password")


@handle_error
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate admin user with comprehensive error handling"""
    try:
        validate_input("Username", username, required=True)
        validate_input("Password", password, required=True)

        password_hash = hash_password(password)

        # Enhanced query with better error handling
        response = supabase.table('admins').select('*').eq('username', username.strip()).execute()

        if not response.data:
            logger.warning(f"No admin found with username: {username}")
            raise AuthenticationError("Invalid credentials")

        stored_user = response.data[0]
        stored_hash = stored_user.get('password_hash', '')

        if not stored_hash:
            logger.error("No password hash found in database")
            raise DatabaseError("User account corrupted - no password hash")

        if stored_hash == password_hash:
            logger.info(f"Successful login for user: {username}")
            return True
        else:
            logger.warning(f"Password mismatch for user: {username}")
            raise AuthenticationError("Invalid credentials")

    except (ValidationError, AuthenticationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected authentication error: {e}")
        raise AuthenticationError("Authentication service temporarily unavailable")


@handle_error
def ensure_admin_exists() -> bool:
    """Create default admin if not exists with proper error handling"""
    try:
        # Check if admin table exists and is accessible
        response = supabase.table('admins').select('username').limit(1).execute()

        # Check for existing admin
        admin_response = supabase.table('admins').select('*').eq('username', 'admin').execute()

        if len(admin_response.data) == 0:
            # Create new admin with the correct password hash
            default_password = 'admin123'
            admin_data = {
                'username': 'admin',
                'password_hash': hash_password(default_password),
                'created_at': datetime.now().isoformat()
            }

            result = supabase.table('admins').insert(admin_data).execute()

            if result.data:
                logger.info("Default admin created successfully")
                st.success("✅ Default admin created successfully!")
                st.info(f"Username: admin | Password: {default_password}")
                return True
            else:
                raise DatabaseError("Failed to create admin account")

        logger.info("Admin account already exists")
        return False

    except Exception as e:
        logger.error(f"Admin creation failed: {e}")
        raise DatabaseError(f"Could not create/verify admin account: {str(e)}")


@handle_error
def get_all_members() -> pd.DataFrame:
    """Fetch all members from database with error handling"""
    try:
        response = supabase.table('image_data_extraction').select('*').order('created_at', desc=True).execute()

        if response.data is None:
            logger.warning("No data returned from members query")
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        logger.info(f"Successfully fetched {len(df)} members")
        #print(df.columns)
        return df

    except Exception as e:
        logger.error(f"Failed to fetch members: {e}")
        raise DatabaseError(f"Could not retrieve members: {str(e)}")


@handle_error
def add_member(name: str, toa: str, payment: int, mode: str) -> bool:
    """Add new member with comprehensive validation and error handling"""
    try:
        # Input validation
        validate_input("Name", name, required=True, min_length=2)

        if payment <= 0:
            raise ValidationError("Payment amount must be greater than 0")

        # Prepare member data
        member_data = {
            'name': name.strip(),
            'toa': toa.strip() if toa else '',
            'payment': float(payment),
            'mode': mode.strip(),
            # 'created_at': datetime.now().isoformat()
        }

        # Check for duplicate names (optional warning)
        existing_response = supabase.table('image_data_extraction').select('name').eq('name', name.strip()).execute()
        if existing_response.data:
            st.warning(f"⚠️ A member named '{name}' already exists. Adding anyway...")

        # Insert member
        response = supabase.table('image_data_extraction').insert(member_data).execute()

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
def update_member(member_id: int, name: str, toa: str, payment: int, mode: str) -> bool:
    """Update member with validation and error handling"""
    try:
        # Input validation
        validate_input("Name", name, required=True, min_length=2)

        if payment <= 0:
            raise ValidationError("Payment amount must be greater than 0")

        # Check if member exists
        check_response = supabase.table('image_data_extraction').select('id').eq('id', member_id).execute()
        if not check_response.data:
            raise ValidationError(f"Member with ID {member_id} not found")

        # Prepare update data - keep payment as integer for BIGINT column
        member_data = {
            'name': name.strip(),
            'payment': payment,  # ✅ Keep as integer for BIGINT column
            'toa': toa.strip() if toa else '',
            'mode': mode.strip(),
            'created_at': datetime.now().isoformat()  # ✅ Changed to updated_at instead of created_at
        }

        # Update member
        response = supabase.table('image_data_extraction').update(member_data).eq('id', member_id).execute()

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
def delete_member(member_id: int) -> bool:
    """Delete member with error handling"""
    try:
        # Check if member exists
        check_response = supabase.table('image_data_extraction').select('id, name').eq('id', member_id).execute()
        if not check_response.data:
            raise ValidationError(f"Member with ID {member_id} not found")

        member_name = check_response.data[0]['name']

        # Delete member
        response = supabase.table('image_data_extraction').delete().eq('id', member_id).execute()

        if response.data is not None:  # Supabase delete returns data on success
            logger.info(f"Successfully deleted member: {member_name} (ID: {member_id})")
            return True
        else:
            raise DatabaseError("Failed to delete member")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting member: {e}")
        raise DatabaseError(f"Could not delete member: {str(e)}")


def get_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate member statistics with error handling"""
    try:
        if df.empty:
            return {'total_members': 0, 'total_payment': 0}

        # Ensure payment column exists and is numeric
        if 'payment' not in df.columns:
            logger.warning("Payment column missing from dataframe")
            return {'total_members': len(df), 'total_payment': 0}

        # Handle non-numeric payment values
        numeric_payments = pd.to_numeric(df['payment'], errors='coerce').fillna(0)

        return {
            'total_members': len(df),
            'total_payment': numeric_payments.sum(),
        }
    except Exception as e:
        logger.error(f"Stats calculation error: {e}")
        return {'total_members': 0, 'total_payment': 0}


@handle_error
def reset_admin_password() -> bool:
    """Reset admin password to default with proper hashing"""
    try:
        default_password = 'admin123'
        new_hash = hash_password(default_password)

        # First, delete any existing admin to avoid conflicts
        supabase.table('admins').delete().eq('username', 'admin').execute()

        # Create fresh admin with correct hash
        admin_data = {
            'username': 'admin',
            'password_hash': new_hash,
            'created_at': datetime.now().isoformat()
        }

        create_response = supabase.table('admins').insert(admin_data).execute()

        if create_response.data:
            logger.info("Admin password reset successfully")
            st.success("✅ Admin account reset successfully!")
            st.info("Username: admin | Password: admin123")
            return True
        else:
            raise DatabaseError("Failed to create admin account")

    except Exception as e:
        logger.error(f"Password reset failed: {e}")
        raise DatabaseError(f"Could not reset password: {str(e)}")


def init_session_state():
    """Initialize session state variables with error recovery"""
    try:
        default_states = {
            'logged_in': False,
            'show_add_form': False,
            'edit_member': None,
            'delete_confirmation': None,
            'search_query': "",
            'error_count': 0,
            'last_error': None,
            'selected_dates_calendar': set(),
            'amount': 0.0,
            'selected_thursdays': []
        }
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    except Exception as e:
        logger.error(f"Session state initialization failed: {e}")
        # Force reset session state on critical failure
        for key in list(st.session_state.keys()):
            del st.session_state[key]


# Session state management with error recovery
def show_error_details(error_message: str):
    """Show detailed error information for debugging"""
    with st.expander("🔧 Error Details (Click to expand)"):
        st.error(error_message)
        st.code(f"""
        Error Count: {st.session_state.get('error_count', 0)}
        Last Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Troubleshooting Tips:
        1. Refresh the page and try again
        2. Check your internet connection
        3. Verify database tables exist in Supabase
        4. Clear browser cache if issues persist
        """)


def process_monthly_payment_data(raw_data, selected_thursdays, weekly_amount):
    """Process payment data and calculate monthly totals with pending amounts"""
    try:
        # Aggregate payments by member name for the entire month
        member_monthly_totals = {}

        for record in raw_data:
            name = record.get('name', 'Unknown')
            payment = record.get('payment', 0)

            # Convert payment to float safely
            try:
                payment_amount = float(payment) if payment is not None else 0.0
            except (ValueError, TypeError):
                payment_amount = 0.0

            # Sum up all payments for each member
            if name in member_monthly_totals:
                member_monthly_totals[name] += payment_amount
            else:
                member_monthly_totals[name] = payment_amount

        # Calculate expected amount based on selected Thursdays
        expected_monthly_amount = len(selected_thursdays) * weekly_amount

        # Create processed data with pending calculations
        processed_data = []
        for name, total_paid in member_monthly_totals.items():
            pending_amount = expected_monthly_amount - total_paid

            processed_data.append({
                "name": name,
                "total_paid": total_paid,
                "expected_amount": expected_monthly_amount,
                "pending_amount": max(0, pending_amount),  # Don't show negative pending
                "overpaid_amount": max(0, -pending_amount),  # Show overpayment if any
                "payment_status": "Complete" if pending_amount <= 0 else "Pending"
            })

        return processed_data

    except Exception as e:
        st.error(f"Error processing monthly payment data: {str(e)}")
        return []


def display_monthly_payment_summary(selected_thursdays, weekly_amount):
    """Display monthly payment summary with totals and pending amounts"""
    try:
        if 'individuals_data' not in st.session_state or not st.session_state.individuals_data:
            st.info("No payment data loaded. Click 'Load Payment Data' to fetch from database.")
            return

        individuals = st.session_state.individuals_data
        total_weeks = len(selected_thursdays)
        expected_total = total_weeks * weekly_amount

        st.write("---")
        st.write(f"### Monthly Payment Summary for {total_weeks} Thursdays")
        st.info(f"Expected payment per person: Rs. {expected_total:,.2f}")

        # Create summary table
        summary_data = []
        for individual in individuals:
            name = individual["name"]
            total_paid = individual["total_paid"]
            pending = individual["pending_amount"]
            overpaid = individual["overpaid_amount"]

            if pending == 0 and overpaid == 0:
                status = "✅ Complete"
                pending_display = "Rs. 0.00"
            elif pending > 0:
                status = f"⚠️ Pending"
                pending_display = f"Rs. {pending:,.2f}"
            else:
                status = f"🔵 Overpaid"
                pending_display = f"Rs. 0.00 (Overpaid: Rs. {overpaid:,.2f})"

            summary_data.append({
                "Name": name,
                "Total Paid": f"Rs. {total_paid:,.2f}",
                "Expected": f"Rs. {expected_total:,.2f}",
                "Pending/Status": pending_display,
                "Status": status
            })

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Overall statistics
        st.write("---")
        st.write("#### 📈 Overall Statistics")

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        total_expected = len(individuals) * expected_total
        total_collected = sum([ind["total_paid"] for ind in individuals])
        total_pending = sum([ind["pending_amount"] for ind in individuals])
        fully_paid_count = sum([1 for ind in individuals if ind["pending_amount"] == 0])

        with col_stat1:
            st.write("Total Members", len(individuals))
        with col_stat2:
            st.write("Total Expected", f"Rs. {total_expected:,.2f}")
        with col_stat3:
            st.write("Total Collected", f"Rs. {total_collected:,.2f}")
        with col_stat4:
            st.write("Total Pending", f"Rs. {total_pending:,.2f}")

        # Collection rate
        #col_rate1, col_rate2 = st.columns(2)
        #with col_rate1:
        #    collection_rate = (total_collected / total_expected * 100) if total_expected > 0 else 0
        #    st.metric("Collection Rate", f"{collection_rate:.1f}%")
        #with col_rate2:
        #    st.metric("Fully Paid Members", f"{fully_paid_count}/{len(individuals)}")

        # Progress bar
        #st.progress(min(collection_rate / 100, 1.0))

        # Export functionality
        if st.button("📥 Export Monthly Summary to CSV"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"monthly_payment_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error displaying monthly payment summary: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def show_calendar_widget():
    """Show calendar widget for date selection with amount calculation"""
    try:
        st.write("### 📅 Calendar Date Selection & Amount Calculator")

        # Initialize session states
        if 'selected_dates_calendar' not in st.session_state:
            st.session_state.selected_dates_calendar = set()
        if 'weekly_amount' not in st.session_state:
            st.session_state.weekly_amount = 1400.0
        if 'calculation_results' not in st.session_state:
            st.session_state.calculation_results = {}
        if 'calendar_initialized' not in st.session_state:
            st.session_state.calendar_initialized = False

        # Amount input section
        col_amount, col_info = st.columns([1, 2])

        with col_amount:
            weekly_amount = st.number_input(
                "Weekly Amount (Rs.)",
                min_value=0.0,
                value=st.session_state.weekly_amount,
                step=50.0,
                format="%.2f",
                help="Enter the amount each person pays per week"
            )
            st.session_state.weekly_amount = weekly_amount

        with col_info:
            st.info("💡 This is the amount each individual pays every Thursday")

        st.write("Select dates (Thursdays are auto-selected, click to deselect):")

        # Date selection controls
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            year = st.selectbox("Year", range(2020, 2030), index=5, key="calendar_year")  # Default to 2025
        with col2:
            month = st.selectbox("Month", range(1, 13), index=datetime.now().month - 1, key="calendar_month")
        with col3:
            if st.button("Clear All"):
                st.session_state.selected_dates_calendar.clear()
                st.session_state.calendar_initialized = False
                st.rerun()
        with col4:
            if st.button("Select All Thursdays"):
                # Auto-select all Thursdays in the month
                thursdays = get_thursdays_in_month(year, month)
                st.session_state.selected_dates_calendar.update(thursdays)
                st.rerun()

        # Auto-select all Thursdays on first load or month/year change
        thursdays_in_month = get_thursdays_in_month(year, month)

        # Check if month/year changed or first time initialization
        current_month_key = f"{year}_{month}"
        if 'last_selected_month' not in st.session_state:
            st.session_state.last_selected_month = current_month_key
            st.session_state.selected_dates_calendar.update(thursdays_in_month)
        elif st.session_state.last_selected_month != current_month_key:
            # Month/year changed - auto-select new month's Thursdays
            st.session_state.last_selected_month = current_month_key
            # Clear previous selections and add new month's Thursdays
            st.session_state.selected_dates_calendar = set(thursdays_in_month)
            st.rerun()

        # Create calendar grid
        cal = calendar.monthcalendar(year, month)
        month_name = calendar.month_name[month]

        st.subheader(f"{month_name} {year}")

        # Display Thursday count for current month
        st.write(f"**Thursdays in {month_name} {year}:** {len(thursdays_in_month)}")

        # Create buttons for each day
        for week_idx, week in enumerate(cal):
            cols = st.columns(7)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # Show day headers for the first week
            if week_idx == 0:
                for i, day_name in enumerate(day_names):
                    if day_name == 'Thu':
                        cols[i].markdown(f"**<span style='color: #ff6b6b;'>{day_name}</span>**", unsafe_allow_html=True)
                    else:
                        cols[i].markdown(f"**{day_name}**")

            for i, day in enumerate(week):
                if day == 0:  # Empty cell
                    cols[i].write("")
                else:
                    current_date = date(year, month, day)
                    date_str = current_date.strftime('%Y-%m-%d')
                    is_thursday = current_date.weekday() == 3  # Thursday = 3
                    is_selected = current_date in st.session_state.selected_dates_calendar

                    # Style button based on day type and selection
                    if is_thursday:
                        if is_selected:
                            button_label = f"🟢 **{day}**"  # Selected Thursday
                        else:
                            button_label = f"🔴 **{day}**"  # Unselected Thursday
                    else:
                        button_label = f"**{day}**" if is_selected else str(day)

                    help_text = f"Thursday - Click to toggle {date_str}" if is_thursday else f"Click to toggle {date_str}"

                    if cols[i].button(button_label, key=f"cal_{date_str}_{week_idx}",
                                      help=help_text):
                        if current_date in st.session_state.selected_dates_calendar:
                            st.session_state.selected_dates_calendar.remove(current_date)
                        else:
                            st.session_state.selected_dates_calendar.add(current_date)
                        st.rerun()

        # Display selected dates and calculations
        if st.session_state.selected_dates_calendar:
            sorted_dates = sorted(list(st.session_state.selected_dates_calendar))
            selected_thursdays = [d for d in sorted_dates if d.weekday() == 3]

            st.write("---")
            st.write("#### 📊 Selection Summary")

            col_summary1, col_summary2 = st.columns(2)

            with col_summary1:
                st.write(f"**Total Selected Dates:** {len(sorted_dates)}")
                st.write(f"**Selected Thursdays:** {len(selected_thursdays)}")
                if selected_thursdays:
                    st.write("**Thursday Dates:**")
                    for thu in selected_thursdays:
                        st.write(f"  • {thu.strftime('%Y-%m-%d (%A)')}")

            with col_summary2:
                total_amount = len(selected_thursdays) * weekly_amount
                st.metric("Total Amount per Person", f"Rs. {total_amount:,.2f}")
                if len(selected_thursdays) > 0:
                    st.metric("Weekly Amount", f"Rs. {weekly_amount:,.2f}")
                    st.metric("Number of Weeks", len(selected_thursdays))

        # Payment tracking section
        if st.session_state.selected_dates_calendar:
            st.write("---")
            st.write("#### 💳 Payment Tracking")

            # Get selected thursdays for filtering
            sorted_dates = sorted(list(st.session_state.selected_dates_calendar))
            selected_thursdays = [d for d in sorted_dates if d.weekday() == 3]

            col_load1, col_load2 = st.columns([1, 2])

            with col_load1:
                if st.button("📊 Load Payment Data"):
                    with st.spinner("Fetching data from Supabase..."):
                        try:
                            # Initialize Supabase client
                            supabase = init_supabase()

                            if supabase and selected_thursdays:
                                # Get selected_year and selected_month from selected thursdays
                                selected_year = selected_thursdays[0].year
                                selected_month = selected_thursdays[0].month
                                month_name_selected = calendar.month_name[selected_month]

                                # Create date range for the selected month only
                                start_date = f"{selected_year}-{selected_month:02d}-01"
                                if selected_month == 12:
                                    end_date = f"{selected_year + 1}-01-01"
                                else:
                                    end_date = f"{selected_year}-{selected_month + 1:02d}-01"

                               # st.info(f"🔍 Filtering data for: **{month_name_selected} {selected_year}**")

                                # Filter records by created_at timestamp
                                response = supabase.table('image_data_extraction') \
                                    .select('name, payment, created_at') \
                                    .gte('created_at', start_date) \
                                    .lt('created_at', end_date) \
                                    .execute()

                                if response.data:
                                    # Process the data with monthly aggregation
                                    payment_data = process_monthly_payment_data(
                                        response.data,
                                        selected_thursdays,
                                        weekly_amount
                                    )
                                    st.session_state.individuals_data = payment_data
                                    #st.success(
                                    #    f"✅ Loaded {month_name_selected} {selected_year} data for {len(payment_data)} individuals")
                                    st.info(
                                        f"📊 Found {len(response.data)} total records for {month_name_selected} {selected_year}")
                                else:
                                    st.warning(f"⚠️ No data found for {month_name_selected} {selected_year}")
                            else:
                                if not supabase:
                                    st.error("❌ Failed to connect to Supabase")
                                else:
                                    st.error("❌ Please select at least one Thursday first")

                        except Exception as e:
                            st.error(f"❌ Error loading data: {str(e)}")

            with col_load2:
                #st.info("💡 This will fetch data ONLY for the selected month using created_at timestamp")
                if selected_thursdays:
                    selected_month_name = calendar.month_name[selected_thursdays[0].month]
                    selected_year_display = selected_thursdays[0].year
                    #st.caption(f"📅 Will filter for: **{selected_month_name} {selected_year_display}**")

            # Display payment summary if data is loaded
            if 'individuals_data' in st.session_state and st.session_state.individuals_data:
                display_monthly_payment_summary(selected_thursdays, weekly_amount)

    except Exception as e:
        st.error(f"Calendar widget error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def get_thursdays_in_month(year, month):
    """Get all Thursdays in a given month"""
    thursdays = []
    cal = calendar.monthcalendar(year, month)

    for week in cal:
        if week[3] != 0:  # Thursday is index 3, check if it's not empty
            thursdays.append(date(year, month, week[3]))

    return thursdays


@handle_error
def search_members(query: str) -> pd.DataFrame:
    """Search members with error handling"""
    try:
        validate_input("Search query", query, required=True, min_length=1)

        # Sanitize search query
        clean_query = query.strip().replace("'", "''")  # Basic SQL injection prevention

        response = supabase.table('image_data_extraction').select('*').or_(

            f'name.ilike.%{clean_query}%,toa.ilike.%{clean_query}%'
        ).order('created_at', desc=True).execute()

        df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        logger.info(f"Search for '{query}' returned {len(df)} results")
        return df

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise DatabaseError(f"Search failed: {str(e)}")


def process_supabase_data_with_created_at(raw_data, selected_thursdays):
    """Process raw Supabase data with created_at filtering"""
    try:
        from datetime import datetime

        # Group payments by individual name and week
        payment_by_person_week = {}

        st.write(f"🔍 **Processing {len(raw_data)} records from Supabase...**")

        for record in raw_data:
            name = record.get('name', 'Unknown')
            payment = record.get('payment', 0)
            created_at_str = record.get('created_at', '')

            # Convert payment to float
            try:
                payment_amount = float(payment) if payment is not None else 0.0
            except (ValueError, TypeError):
                payment_amount = 0.0

            # Parse created_at timestamp - FIXED TO HANDLE MULTIPLE FORMATS
            try:
                if created_at_str:
                    record_date = None

                    # Try different date formats
                    formats_to_try = [
                        '%Y-%m-%d',  # 2025-09-06
                        '%Y-%m-%dT%H:%M:%S',  # 2025-09-06T10:30:45
                        '%Y-%m-%dT%H:%M:%SZ',  # 2025-09-06T10:30:45Z
                        '%Y-%m-%dT%H:%M:%S.%fZ',  # 2025-09-06T10:30:45.123456Z
                        '%Y-%m-%d %H:%M:%S',  # 2025-09-06 10:30:45
                    ]

                    for date_format in formats_to_try:
                        try:
                            created_at = datetime.strptime(created_at_str, date_format)
                            record_date = created_at.date()
                            break
                        except ValueError:
                            continue

                    # If none of the above formats worked, try isoformat
                    if record_date is None:
                        try:
                            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                            record_date = created_at.date()
                        except ValueError:
                            pass

                    # If we successfully parsed the date
                    if record_date:
                        # Find the closest Thursday (assuming data is uploaded on or near Thursdays)
                        closest_thursday_idx = None
                        min_days_diff = float('inf')

                        for idx, thursday in enumerate(selected_thursdays):
                            days_diff = abs((record_date - thursday).days)
                            if days_diff < min_days_diff:
                                min_days_diff = days_diff
                                closest_thursday_idx = idx

                        # Include if within 7 days of a Thursday (more flexible range)
                        if closest_thursday_idx is not None and min_days_diff <= 7:
                            week_key = closest_thursday_idx

                            if name not in payment_by_person_week:
                                payment_by_person_week[name] = {}

                            # Sum payments for the same person in the same week
                            if week_key in payment_by_person_week[name]:
                                payment_by_person_week[name][week_key] += payment_amount
                            else:
                                payment_by_person_week[name][week_key] = payment_amount

                            # Debug info
                            # st.write(
                            # f"✅ Matched: {name} - Rs.{payment_amount} on {record_date} → Thursday {selected_thursdays[week_key].strftime('%Y-%m-%d')}")
                        else:
                            st.write(
                                f"⚠️ Skipped: {name} - Rs.{payment_amount} on {record_date} (too far from selected Thursdays)")
                    else:
                        st.warning(f"❌ Could not parse date format: '{created_at_str}' for {name}")

            except Exception as e:
                st.warning(f"❌ Error processing record for {name}: {str(e)}")
                continue

        # Convert to the required format
        processed_data = []
        for name, week_payments in payment_by_person_week.items():
            payments = []
            for i in range(len(selected_thursdays)):
                payment = week_payments.get(i, 0.0)
                payments.append(payment)

            processed_data.append({
                "name": name,
                "payments": payments
            })

        st.write(f"📊 **Successfully processed {len(processed_data)} individuals**")
        return processed_data

    except Exception as e:
        st.error(f"Error processing Supabase data with created_at: {str(e)}")
        return []


def display_payment_summary(selected_thursdays, weekly_amount):
    """Display payment summary for all individuals"""
    try:
        if 'individuals_data' not in st.session_state or not st.session_state.individuals_data:
            st.info("No payment data loaded. Click 'Load Payment Data' to fetch from database.")
            return

        individuals = st.session_state.individuals_data
        total_weeks = len(selected_thursdays)
        expected_total = total_weeks * weekly_amount

        st.write(f"**Payment Summary for {total_weeks} selected Thursdays:**")
        # st.write(f"**Data Source:** Supabase table 'image_data_extraction'")

        # Create summary table
        summary_data = []
        for individual in individuals:
            payments = individual["payments"]
            total_paid = sum(payments)
            pending = expected_total - total_paid

            if pending == 0:
                status = "✅ Complete"
            elif pending > 0:
                status = f"⚠️ Pending: Rs. {pending:,.2f}"
            else:
                status = f"🔵 Overpaid: Rs. {abs(pending):,.2f}"

            summary_data.append({
                "Name": individual["name"],
                "Expected": f"Rs. {expected_total:,.2f}",
                "Paid": f"Rs. {total_paid:,.2f}",
                "Pending": f"Rs. {pending:,.2f}",
                # "Payment %": f"{payment_percentage:.1f}%",
                "Status": status
            })

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detailed individual breakdown
        with st.expander("📋 Detailed Payment Breakdown"):
            for individual in individuals:
                st.write(f"**{individual['name']}:**")
                payments_df = pd.DataFrame({
                    'Thursday': [d.strftime('%Y-%m-%d') for d in selected_thursdays],
                    'Amount Paid': [f"Rs. {p:,.2f}" for p in individual['payments']],
                    'Status': ['✅' if p > 0 else '❌' for p in individual['payments']]
                })
                st.dataframe(payments_df, use_container_width=True, hide_index=True)
                st.write("---")

        # Overall statistics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        total_expected = len(individuals) * expected_total
        total_collected = sum([sum(ind["payments"]) for ind in individuals])
        total_pending = total_expected - total_collected
        fully_paid_count = sum([1 for ind in individuals if sum(ind["payments"]) >= expected_total])

        with col_stat1:
            st.metric("Total Expected", f"Rs. {total_expected:,.2f}")
        with col_stat2:
            st.metric("Total Collected", f"Rs. {total_collected:,.2f}")
        with col_stat3:
            st.metric("Total Pending", f"Rs. {total_pending:,.2f}")
        with col_stat4:
            st.metric("Fully Paid", f"{fully_paid_count}/{len(individuals)}")

        # Collection rate
        collection_rate = (total_collected / total_expected) * 100 if total_expected > 0 else 0
        # st.progress(collection_rate / 100)
        # st.write(f"**Collection Rate:** {collection_rate:.1f}%")

        # Export functionality
        if st.button("📥 Export Summary to CSV"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"payment_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error displaying payment summary: {str(e)}")



@handle_error
def get_available_dates_from_db(year: int, month: int) -> list:
    """Get all unique dates from database for given year and month"""
    try:
        # Create date range for the selected month
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"

        # Query database for records in the selected month
        response = supabase.table('image_data_extraction') \
            .select('created_at') \
            .gte('created_at', start_date) \
            .lt('created_at', end_date) \
            .execute()

        if response.data:
            # Extract unique dates
            dates_set = set()
            for record in response.data:
                try:
                    # Parse the created_at timestamp
                    created_at_str = record.get('created_at', '')
                    if created_at_str:
                        # Handle different date formats
                        date_obj = None
                        formats_to_try = [
                            '%Y-%m-%d',
                            '%Y-%m-%dT%H:%M:%S',
                            '%Y-%m-%dT%H:%M:%SZ',
                            '%Y-%m-%dT%H:%M:%S.%fZ',
                            '%Y-%m-%d %H:%M:%S',
                        ]

                        for date_format in formats_to_try:
                            try:
                                date_obj = datetime.strptime(created_at_str, date_format).date()
                                break
                            except ValueError:
                                continue

                        # Try isoformat if other formats failed
                        if date_obj is None:
                            try:
                                date_obj = datetime.fromisoformat(created_at_str.replace('Z', '+00:00')).date()
                            except ValueError:
                                continue

                        if date_obj:
                            dates_set.add(date_obj)

                except Exception as e:
                    logger.warning(f"Error parsing date {record.get('created_at', '')}: {e}")
                    continue

            # Convert to sorted list
            available_dates = sorted(list(dates_set))
            return available_dates
        else:
            return []

    except Exception as e:
        logger.error(f"Error fetching available dates: {e}")
        raise DatabaseError(f"Could not fetch available dates: {str(e)}")


@handle_error
def get_members_by_date(selected_date) -> pd.DataFrame:
    """Get all members created on a specific date"""
    try:
        # Convert date to string format for database query
        date_str = selected_date.strftime('%Y-%m-%d')
        next_date_str = (selected_date + timedelta(days=1)).strftime('%Y-%m-%d')

        # Query database for records on the selected date
        response = supabase.table('image_data_extraction') \
            .select('*') \
            .gte('created_at', date_str) \
            .lt('created_at', next_date_str) \
            .order('created_at', desc=True) \
            .execute()

        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Found {len(df)} records for date {date_str}")
            return df
        else:
            logger.info(f"No records found for date {date_str}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error fetching members by date: {e}")
        raise DatabaseError(f"Could not fetch members for date: {str(e)}")


def show_view_records():
    """Show view records interface with date filtering"""
    st.header("View Records by Date")

    # Back button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("⬅️ Back to Dashboard", key="back_from_view_records"):
            st.session_state.show_view_records = False
            st.rerun()

    st.write("Select a month and year to view available record dates:")

    # Month and Year selection
    col_month, col_year = st.columns(2)

    with col_month:
        selected_month = st.selectbox(
            "Month",
            range(1, 13),
            format_func=lambda x: calendar.month_name[x],
            index=datetime.now().month - 1,
            key="view_records_month"
        )

    with col_year:
        current_year = datetime.now().year
        selected_year = st.selectbox(
            "Year",
            range(current_year - 5, current_year + 2),
            index=5,  # Default to current year
            key="view_records_year"
        )

    # Load available dates button
    if st.button("🔍 Load Available Dates", type="primary"):
        with st.spinner(f"Loading dates for {calendar.month_name[selected_month]} {selected_year}..."):
            try:
                available_dates = get_available_dates_from_db(selected_year, selected_month)
                st.session_state.available_dates = available_dates
                st.session_state.selected_month_year = f"{calendar.month_name[selected_month]} {selected_year}"

                if available_dates:
                    st.success(f"Found records on {len(available_dates)} different dates")
                else:
                    st.info(f"No records found for {calendar.month_name[selected_month]} {selected_year}")

            except Exception as e:
                st.error(f"Failed to load dates: {str(e)}")

    # Show date selection dropdown if dates are available
    if 'available_dates' in st.session_state and st.session_state.available_dates:
        #st.write("---")
        st.write(f"**Available dates in {st.session_state.get('selected_month_year', 'Selected Month')}:**")

        # Date selection dropdown
        selected_date = st.selectbox(
            "Select Date",
            st.session_state.available_dates,
            format_func=lambda x: x.strftime('%Y-%m-%d (%A)'),
            key="selected_view_date"
        )

        # Show records for selected date
        if st.button("📊 View Records", type="primary"):
            with st.spinner(f"Loading records for {selected_date.strftime('%Y-%m-%d')}..."):
                try:
                    st.session_state.filtered_date_members = get_members_by_date(selected_date)
                    st.session_state.current_filter_date = selected_date
                    st.success(f"Loaded records for {selected_date.strftime('%Y-%m-%d')}")
                except Exception as e:
                    st.error(f"Failed to load records: {str(e)}")

    # Display filtered members using show_members_list() logic
    if 'filtered_date_members' in st.session_state and not st.session_state.filtered_date_members.empty:
        st.write("---")
        filter_date = st.session_state.get('current_filter_date')
        if filter_date:
            st.subheader(f"📅 Records for {filter_date.strftime('%Y-%m-%d (%A)')}")

        # Use the existing show_members_list logic with filtered data
        show_filtered_members_list(st.session_state.filtered_date_members)


def show_filtered_members_list(df):
    """Display members list in editable table format"""
    try:
        if df is None or df.empty:
            st.info("No records found for the selected date.")
            return

        # Show statistics
        stats = get_stats(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Members", stats['total_members'])
        with col2:
            st.metric("Total Payment", f"Rs. {stats['total_payment']:,.2f}")

        st.markdown("---")
        st.subheader("Edit Records")
        st.info("You can edit multiple records directly in the table below. Click 'Save All Changes' when done.")

        # Prepare dataframe for editing
        edit_df = df.copy()

        # Format payment for display
        if 'payment' in edit_df.columns:
            edit_df['payment'] = edit_df['payment'].apply(
                lambda x: float(x) if pd.notnull(x) else 0.0
            )

        # Format created_at for display
        if 'created_at' in edit_df.columns:
            edit_df['created_at_display'] = pd.to_datetime(
                edit_df['created_at'], errors='coerce'
            ).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Select and reorder columns for display
        display_columns = ['id', 'name', 'toa', 'payment', 'mode']
        available_columns = [col for col in display_columns if col in edit_df.columns]

        edit_df_display = edit_df[available_columns].copy()

        # Rename columns for better display
        column_config = {
            'id': st.column_config.NumberColumn('ID', disabled=True, help="Record ID (cannot be edited)"),
            'name': st.column_config.TextColumn('Name', required=True, max_chars=100),
            'toa': st.column_config.TextColumn('Time of Arrival', max_chars=50),
            'payment': st.column_config.NumberColumn('Payment (Rs.)', required=True, min_value=0, format="%.2f"),
            'mode': st.column_config.TextColumn('Mode', max_chars=50),
            'created_at_display': st.column_config.TextColumn('Created At', disabled=True,
                                                              help="Creation date (cannot be edited)")
        }

        # Add checkbox column for selection
        edit_df_display.insert(0, 'Select', False)

        # Update column config to include checkbox
        column_config['Select'] = st.column_config.CheckboxColumn(
            'Select',
            help="Check to select for deletion",
            default=False
        )

        # Show editable dataframe
        edited_df = st.data_editor(
            edit_df_display,
            use_container_width=True,
            num_rows="fixed",
            column_config=column_config,
            hide_index=True,
            key="filtered_members_editor"
        )

        st.markdown("---")

        # Action buttons
        col_save, col_delete, col_cancel = st.columns([1, 1, 2])

        with col_save:
            if st.button("💾 Save All Changes", type="primary", use_container_width=True):
                with st.spinner("Saving changes..."):
                    try:
                        success_count = 0
                        error_count = 0

                        # Compare original and edited dataframes
                        for idx in range(len(edited_df)):
                            original_row = edit_df_display.iloc[idx]
                            edited_row = edited_df.iloc[idx]

                            # Check if any changes were made
                            if not original_row.equals(edited_row):
                                record_id = int(edited_row['id'])

                                # Prepare update data
                                update_data = {
                                    'name': str(edited_row['name']).strip(),
                                    'toa': str(edited_row['toa']).strip() if pd.notnull(edited_row['toa']) else '',
                                    'payment': int(edited_row['payment']),
                                    'mode': str(edited_row['mode']).strip() if pd.notnull(edited_row['mode']) else '',
                                }

                                # Update in database
                                response = supabase.table('image_data_extraction').update(update_data).eq('id',
                                                                                                          record_id).execute()

                                if response.data:
                                    success_count += 1
                                else:
                                    error_count += 1

                        if success_count > 0:
                            st.success(f"Successfully updated {success_count} record(s)!")
                            time.sleep(1)
                            st.rerun()
                        elif error_count > 0:
                            st.error(f"Failed to update {error_count} record(s)")
                        else:
                            st.info("No changes detected")

                    except Exception as e:
                        st.error(f"Error saving changes: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

        with col_delete:
            # Count selected rows
            selected_rows = edited_df[edited_df['Select'] == True]
            num_selected = len(selected_rows)

            if num_selected > 0:
                st.warning(f"⚠️ {num_selected} record(s) selected for deletion")

                if st.button(f"🗑️ Delete {num_selected} Selected", type="secondary", use_container_width=True):
                    with st.spinner(f"Deleting {num_selected} record(s)..."):
                        try:
                            deleted_count = 0
                            failed_count = 0

                            for _, row in selected_rows.iterrows():
                                record_id = int(row['id'])
                                record_name = row['name']
                                try:
                                    # Delete from database
                                    response = supabase.table('image_data_extraction').delete().eq('id',
                                                                                                   record_id).execute()

                                    # Check if delete was successful
                                    if response.data and len(response.data) > 0:
                                        deleted_count += 1
                                    else:
                                        failed_count += 1
                                        st.warning(f"Failed to delete {record_name} (ID: {record_id})")
                                except Exception as delete_error:
                                    failed_count += 1
                                    st.error(f"Error deleting {record_name}: {str(delete_error)}")

                            # Show results
                            if deleted_count > 0:
                                st.success(f"✅ Successfully deleted {deleted_count} record(s)!")
                                if failed_count > 0:
                                    st.warning(f"⚠️ Failed to delete {failed_count} record(s)")
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete any records")

                        except Exception as e:
                            st.error(f"Error during deletion: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
            else:
                st.info("Select rows using the checkbox column to delete")
                st.button("🗑️ Delete Selected", disabled=True, use_container_width=True)

        with col_cancel:
            if st.button("↩️ Back to View Records", use_container_width=True):
                st.session_state.show_view_records = True
                st.rerun()

    except Exception as e:
        st.error("Error displaying filtered members")
        show_error_details(str(e))
        import traceback
        st.error(traceback.format_exc())

def show_sidebar():
    """Show sidebar with file upload and other options - UPDATED VERSION"""
    try:
        # Force sidebar to be visible
        st.sidebar.markdown("")  # This ensures sidebar is created

        with st.sidebar:
            # Error monitoring
            if st.session_state.get('error_count', 0) > 0:
                st.warning(f"⚠️ {st.session_state.error_count} error(s) occurred this session")

            # IMAGE PROCESSING SECTION (MOVED FROM show_add_member_form)
            st.header("📂 Image Processing")

            # Option selector for upload or capture
            capture_option = st.radio(
                "Choose input method:",
                options=["Upload File", "Capture Image"],
                key="input_method"
            )

            if capture_option == "Upload File":
                # File upload option
                uploaded_files = st.file_uploader(
                    "Upload image file(s)",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    help="Select one or more image files from your device"
                )

                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")

                    # Store in session state
                    st.session_state.image_to_process = uploaded_files
                    st.session_state.image_source = "upload"

            else:  # Capture Image
                # Camera capture option
                st.info("📷 Use camera to capture")
                captured_image = st.camera_input(
                    "Take a picture",
                    help="Click to take a picture using your device's camera"
                )

                if captured_image is not None:
                    st.success("✅ Image captured!")
                    # Store in session state for dashboard processing
                    st.session_state.image_to_process = captured_image
                    st.session_state.image_source = "capture"

            # Process button
            if st.session_state.get('image_to_process') is not None:
                if st.button("🔄 Process Image", use_container_width=True, type="primary"):
                    with st.spinner("Processing..."):
                        try:
                            # Process the image and extract the data
                            all_data = []

                            # Process single or multiple images
                            images_to_process = st.session_state.image_to_process
                            if not isinstance(images_to_process, list):
                                images_to_process = [images_to_process]

                            for img in images_to_process:
                                try:
                                    data = upload_and_process_image(img)
                                    if isinstance(data, pd.DataFrame):
                                        all_data.append(data)
                                    elif isinstance(data, list):  # In case it returns list of dicts
                                        all_data.append(pd.DataFrame(data))
                                except Exception as e:
                                    st.warning(
                                        f"Failed to process {img.name if hasattr(img, 'name') else 'captured image'}: {e}")

                            # Combine all results
                            if all_data:
                                combined_data = pd.concat(all_data, ignore_index=True)
                                st.session_state.extracted_data = combined_data
                                st.session_state.data_processed = True
                                st.success("✅ All images processed successfully!")
                                st.rerun()  # Refresh to show data in main area
                            else:
                                st.error("No data extracted from uploaded images.")

                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            st.session_state.extracted_data = None
                            st.session_state.data_processed = False

            # Clear processed data button
            if st.session_state.get('data_processed'):
                if st.button("🗑️ Clear Data", use_container_width=True):
                    st.session_state.image_to_process = None
                    st.session_state.extracted_data = None
                    st.session_state.data_processed = False
                    st.rerun()

            st.divider()

            # Action buttons with error handling
            if st.button("➕ Members Meeting Attendance Data", use_container_width=True, key='sidebar_add_member'):
                st.session_state.show_add_form = True
                st.session_state.edit_member = None
                st.session_state.show_view_records = False
                st.session_state.show_calendar = False
                st.session_state.show_personal_data = False
                st.session_state.show_meeting_data = False  # Clear other views
                st.rerun()

            # View Records Button
            if st.button("👀 View Records", use_container_width=True, key="sidebar_view_records"):
                st.session_state.show_view_records = True
                st.session_state.show_add_form = False
                st.session_state.show_calendar = False
                st.session_state.show_personal_data = False
                st.session_state.show_meeting_data = False  # Clear other views
                st.session_state.edit_member = None
                st.rerun()

            # NEW: Meeting Data Button
            if st.button("📊 Meeting Data", use_container_width=True, key="sidebar_meeting_data"):
                st.session_state.show_meeting_data = True
                st.session_state.show_add_form = False
                st.session_state.show_calendar = False
                st.session_state.show_view_records = False
                st.session_state.show_personal_data = False
                st.session_state.edit_member = None
                st.rerun()

            if st.button("📄 Personal Data", use_container_width=True, key="sidebar_personal_data"):
                st.session_state.show_personal_data = True
                st.session_state.show_add_form = False
                st.session_state.show_calendar = False
                st.session_state.show_view_records = False
                st.session_state.show_meeting_data = False  # Clear other views
                st.rerun()

            # Navigation buttons
            if st.button("📅 Calendar", use_container_width=True, key="sidebar_calendar"):
                st.session_state.show_calendar = True
                st.session_state.show_add_form = False
                st.session_state.edit_member = None
                st.session_state.show_view_records = False
                st.session_state.show_personal_data = False
                st.session_state.show_meeting_data = False  # Clear other views
                st.rerun()

            if st.button("📊 Export Data", use_container_width=True, key="sidebar_export"):
                try:
                    df = get_all_members()
                    if df is not None and not df.empty:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"bni_members_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("No data to export")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")

            if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    except Exception as e:
        st.sidebar.error(f"Sidebar error: {str(e)}")
        logger.error(f"Sidebar error: {e}")

def show_meeting_data():
    """Show meeting data interface with month/year selection and payment tracking"""
    st.header("📊 Meeting Data Analysis")

    # Back button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("⬅️ Back to Dashboard", key="back_from_meeting_data"):
            st.session_state.show_meeting_data = False
            st.rerun()

    # Initialize session state for meeting data
    if 'meeting_data_selected_month' not in st.session_state:
        st.session_state.meeting_data_selected_month = datetime.now().month
    if 'meeting_data_selected_year' not in st.session_state:
        st.session_state.meeting_data_selected_year = datetime.now().year
    if 'weekly_payment_amount' not in st.session_state:
        st.session_state.weekly_payment_amount = 1400.0

    st.write("### 📅 Select Month and Year")

    # Month and Year selection
    col_month, col_year, col_amount = st.columns([2, 2, 2])

    with col_month:
        selected_month = st.selectbox(
            "Month",
            range(1, 13),
            format_func=lambda x: calendar.month_name[x],
            index=st.session_state.meeting_data_selected_month - 1,
            key="meeting_data_month"
        )

    with col_year:
        current_year = datetime.now().year
        selected_year = st.selectbox(
            "Year",
            range(current_year - 5, current_year + 2),
            index=current_year - (current_year - 5),
            key="meeting_data_year"
        )

    with col_amount:
        weekly_payment = st.number_input(
            "Weekly Payment Amount (Rs.)",
            min_value=0.0,
            value=st.session_state.weekly_payment_amount,
            step=50.0,
            format="%.2f",
            help="Amount each person should pay per week"
        )
        st.session_state.weekly_payment_amount = weekly_payment

    # Store selections
    st.session_state.meeting_data_selected_month = selected_month
    st.session_state.meeting_data_selected_year = selected_year

    # Load available dates button
    if st.button("🔍 Load Meeting Dates", type="primary"):
        with st.spinner(f"Loading dates for {calendar.month_name[selected_month]} {selected_year}..."):
            try:
                available_dates = get_available_dates_from_db(selected_year, selected_month)
                st.session_state.meeting_available_dates = available_dates
                st.session_state.meeting_selected_month_year = f"{calendar.month_name[selected_month]} {selected_year}"

                if available_dates:
                    st.success(f"Found meeting records on {len(available_dates)} different dates")

                    # Calculate number of Thursdays in the month for reference
                    thursdays_in_month = get_thursdays_in_month(selected_year, selected_month)
                    st.info(
                        f"ℹ️ There are {len(thursdays_in_month)} Thursdays in {calendar.month_name[selected_month]} {selected_year}")
                else:
                    st.info(f"No meeting records found for {calendar.month_name[selected_month]} {selected_year}")

            except Exception as e:
                st.error(f"Failed to load dates: {str(e)}")

    # Show date selection if dates are available
    if 'meeting_available_dates' in st.session_state and st.session_state.meeting_available_dates:
        st.write("---")
        st.write(
            f"**Available meeting dates in {st.session_state.get('meeting_selected_month_year', 'Selected Month')}:**")

        # Date selection dropdown
        selected_date = st.selectbox(
            "Select Meeting Date",
            st.session_state.meeting_available_dates,
            format_func=lambda x: x.strftime('%Y-%m-%d (%A)'),
            key="selected_meeting_date"
        )

        # Show records for selected date
        if st.button("📊 Load Meeting Data", type="primary"):
            with st.spinner(f"Loading meeting data for {selected_date.strftime('%Y-%m-%d')}..."):
                try:
                    # Load data for selected date
                    date_records = get_members_by_date(selected_date)

                    # Load all data for the month to calculate totals
                    month_records = get_members_by_month(selected_year, selected_month)

                    st.session_state.meeting_date_records = date_records
                    st.session_state.meeting_month_records = month_records
                    st.session_state.current_meeting_date = selected_date

                    #st.success(f"Loaded meeting data for {selected_date.strftime('%Y-%m-%d')}")
                except Exception as e:
                    st.error(f"Failed to load meeting data: {str(e)}")

    # Display meeting data if loaded
    if ('meeting_date_records' in st.session_state and
            'meeting_month_records' in st.session_state and
            not st.session_state.meeting_date_records.empty):
        show_meeting_data_analysis_clean()

def show_meeting_data_analysis_clean():
    """Display meeting data analysis with weekly summary - CLEAN VERSION"""
    try:
        date_records = st.session_state.meeting_date_records
        selected_date = st.session_state.current_meeting_date
        weekly_payment = st.session_state.weekly_payment_amount

        st.write("---")
        st.subheader(f"📊 Meeting Data for {selected_date.strftime('%Y-%m-%d (%A)')}")

        # Enhanced payment conversion function
        def safe_payment_convert(payment_value):
            """Safely convert payment value to float with extensive validation"""
            if payment_value is None:
                return 0.0

            if pd.isna(payment_value):
                return 0.0

            payment_str = str(payment_value).strip()

            if not payment_str or payment_str.lower() in ['nan', 'none', 'null', '']:
                return 0.0

            payment_str = payment_str.replace('Rs.', '').replace('₹', '').replace(',', '').strip()

            try:
                result = float(payment_str)
                if math.isnan(result):
                    return 0.0
                return result
            except (ValueError, TypeError):
                return 0.0

        # Table: Records for selected date with payment status
        st.write("#### 📋 Attendance & Payment Records for Selected Week")
        if not date_records.empty:
            weekly_display_data = []
            total_collected_week = 0.0
            total_pending_week = 0.0

            for _, row in date_records.iterrows():
                name = row.get('name', 'Unknown')
                toa = row.get('toa', 'Not specified')
                payment_amount = safe_payment_convert(row.get('payment', 0))
                total_collected_week += payment_amount

                # Calculate pending amount for this week
                pending_amount = weekly_payment - payment_amount
                total_pending_week += max(0, pending_amount)

                if pending_amount <= 0:
                    status = "✅ Complete"
                    pending_display = "Rs. 0.00"
                else:
                    status = f"⚠️ Pending"
                    pending_display = f"Rs. {pending_amount:,.2f}"

                weekly_display_data.append({
                    "Name": name,
                    "Time of Arrival (TOA)": toa,
                    "Payment": f"Rs. {payment_amount:,.2f}",
                    "Pending": pending_display,
                    "Status": status
                })

            weekly_df = pd.DataFrame(weekly_display_data)
            st.dataframe(weekly_df, use_container_width=True, hide_index=True)

            # Weekly statistics
            st.write("#### 📈 Weekly Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Members Present", len(date_records))
            with col2:
                st.write("Total Collected", f"Rs. {total_collected_week:,.2f}")
            with col3:
                st.write("Total Pending", f"Rs. {total_pending_week:,.2f}")


            # Export functionality
            if st.button("📥 Export Weekly Records"):
                csv_data = weekly_df.to_csv(index=False)
                st.download_button(
                    label="Download Weekly Records CSV",
                    data=csv_data,
                    file_name=f"weekly_meeting_data_{selected_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        else:
            st.info("No records found for the selected date.")

    except Exception as e:
        st.error(f"Error displaying meeting data analysis: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

@handle_error
def get_members_by_month(year: int, month: int) -> pd.DataFrame:
    """Get all members for a specific month"""
    try:
        # Create date range for the selected month
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"

        # Query database for records in the selected month
        response = supabase.table('image_data_extraction') \
            .select('*') \
            .gte('created_at', start_date) \
            .lt('created_at', end_date) \
            .order('created_at', desc=True) \
            .execute()

        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Found {len(df)} records for {calendar.month_name[month]} {year}")
            return df
        else:
            logger.info(f"No records found for {calendar.month_name[month]} {year}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error fetching members by month: {e}")
        raise DatabaseError(f"Could not fetch members for month: {str(e)}")


# Solution 4: Quick Debug - Add this to your current main exeution
def debug_sidebar():
    """Quick debug function to force sidebar visibility"""
    with st.sidebar:
        st.write("🔧 Debug: Sidebar is working!")
        st.write("If you see this, sidebar is functional")


# Solution 5: Modified show_dashboard to ensure sidebar is called

def verify_password_hash():
    """Test function to verify password hashing"""
    test_password = "admin123"
    expected_hash = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"
    actual_hash = hash_password(test_password)
    print(f"Expected: {expected_hash}")
    print(f"Actual:   {actual_hash}")
    print(f"Match:    {expected_hash == actual_hash}")
    return expected_hash == actual_hash


# Add this function temporarily for debugging


# Enhanced login page with error recovery
def show_login():
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🛡️ BNI Brilliance Admin Login</h1>
        <p>Enter your credentials to access the member management system</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Show system status
        try:
            # Quick health check
            health_check = supabase.table('image_data_extraction').select('id').limit(1).execute()
            st.success("🟢 Database connection healthy")
        except Exception as e:
            st.error("🔴 Database connection issues")
            show_error_details(str(e))

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter admin username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                with st.spinner("Authenticating..."):
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.error_count = 0  # Reset error count on success
                        st.success("✅ Login successful!")
                        time.sleep(1)
                        st.rerun()

        st.info("""
        **Default Credentials:**  
        Username: `admin`  
        Password: `admin123`
        """)

        # Emergency reset button
        st.divider()
        st.subheader("🆘 Having trouble logging in?")

        col_reset1, col_reset2 = st.columns(2)

        with col_reset1:
            if st.button("🔄 Reset Admin Password", help="Resets admin password to default"):
                if reset_admin_password():
                    st.success("✅ Admin password reset to 'admin123'")
                    time.sleep(2)
                    st.rerun()

        with col_reset2:
            if st.button("🔧 Create Admin Account", help="Creates admin account if missing"):
                if ensure_admin_exists():
                    time.sleep(1)
                    st.rerun()



# Enhanced dashboard with error monitoring
def show_dashboard():
    """Main dashboard function - UPDATED VERSION"""
    st.title("BNI Brilliance Member Management System")

    # Initialize session state variables (add the new one)
    if "show_calendar" not in st.session_state:
        st.session_state.show_calendar = False

    if "show_add_form" not in st.session_state:
        st.session_state.show_add_form = False

    if "show_view_records" not in st.session_state:
        st.session_state.show_view_records = False

    if "show_meeting_data" not in st.session_state:  # NEW
        st.session_state.show_meeting_data = False

    if "edit_member" not in st.session_state:
        st.session_state.edit_member = None

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    if "selected_dates_calendar" not in st.session_state:
        st.session_state.selected_dates_calendar = set()

    if "delete_confirmation" not in st.session_state:
        st.session_state.delete_confirmation = None

    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None

    if "data_processed" not in st.session_state:
        st.session_state.data_processed = False

    # Initialize session state for extracted data date selection
    if 'extracted_data_selected_date' not in st.session_state:
        st.session_state.extracted_data_selected_date = datetime.now().date()

    # Display extracted data section (only if data exists)
    if st.session_state.get('data_processed') and st.session_state.get('extracted_data') is not None:

        st.header("📝 Extracted Data - Review & Edit")

        # Show source information
        source_info = st.session_state.get('image_source', 'unknown')
        if source_info == "upload":
            st.info("📁 Data extracted from uploaded file")
        else:
            st.info("📷 Data extracted from captured image")

        # DATE SELECTION SECTION - NEW
        st.subheader("📅 Select Date for This Data")
        st.info("Choose the date for which you want to save this attendance data")

        col_date1, col_date2, col_date3 = st.columns(3)

        with col_date1:
            selected_year = st.selectbox(
                "Year",
                range(datetime.now().year - 2, datetime.now().year + 2),
                index=2,  # Current year
                key="extracted_data_year"
            )

        with col_date2:
            selected_month = st.selectbox(
                "Month",
                range(1, 13),
                format_func=lambda x: calendar.month_name[x],
                index=datetime.now().month - 1,  # Current month
                key="extracted_data_month"
            )

        with col_date3:
            # Get number of days in selected month
            days_in_month = calendar.monthrange(selected_year, selected_month)[1]
            selected_day = st.selectbox(
                "Day",
                range(1, days_in_month + 1),
                index=min(datetime.now().day - 1, days_in_month - 1),  # Current day or last day of month
                key="extracted_data_day"
            )

        # Construct the selected date
        selected_date1 = date(selected_year, selected_month, selected_day)
        st.session_state.extracted_data_selected_date = selected_date1

        # Display selected date prominently
        st.success(f"📌 Data will be saved for: **{selected_date1.strftime('%A, %B %d, %Y')}**")

        st.divider()

        extracted_data = st.session_state.get('extracted_data')

        if extracted_data is None:
            st.info("No data extracted yet. Please upload or capture an image to begin.")
            return

        # Ensure extracted_data is a pandas DataFrame
        if isinstance(extracted_data, list):
            extracted_data = pd.DataFrame(extracted_data)

        # Remove 'id' column if it exists
        if "id" in extracted_data.columns:
            extracted_data = extracted_data.drop(columns=["id"])

        # Display editable dataframe using st.data_editor
        st.write("Edit the data below before saving to database:")

        edited_data = st.data_editor(
            extracted_data,
            num_rows="dynamic",  # Allows adding/deleting rows
            use_container_width=True,
            hide_index=False,
            column_config={
                "Name": st.column_config.TextColumn(
                    "Name",
                    help="Member name",
                    max_chars=100,
                    required=True
                ),
                "TOA": st.column_config.TextColumn(
                    "TOA (Type of Activity)",
                    help="Type of activity",
                    max_chars=50
                ),
                "Payment": st.column_config.TextColumn(
                    "Payment",
                    help="Payment amount or status",
                    max_chars=50
                ),
                "Mode": st.column_config.TextColumn(
                    "Mode",
                    help="Payment mode",
                    max_chars=50
                )
            },
            key="extracted_data_editor"
        )

        # Action buttons
        col1, col2, col3,col4 = st.columns([2, 2, 2,2])

        with col1:
            if st.button("💾 Save to Database", type="primary", key="save_extracted_data"):
                if edited_data is not None and len(edited_data) > 0:
                    with st.spinner(f"Saving data to database for {selected_date1.strftime('%Y-%m-%d')}..."):
                        # Pass the selected date to the save function
                        save_data_to_supabase(edited_data, selected_date=selected_date1)
                        st.success(f"✅ Data saved successfully for {selected_date1.strftime('%Y-%m-%d')}!")
                        # Clear the extracted data after saving
                        st.session_state.extracted_data = None
                        st.session_state.data_processed = False
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("⚠️ No data to save!")

        with col2:
            if st.button("🗑️ Clear Data", key="clear_extracted_data"):
                st.session_state.extracted_data = None
                st.session_state.data_processed = False
                st.rerun()

        with col3:
            if st.button("🏠 Back to Dashboard", key="back_to_dashboard_from_extracted"):
                st.session_state.extracted_data = None
                st.session_state.data_processed = False
                st.session_state.show_calendar = False
                st.session_state.show_add_form = False
                st.session_state.show_view_records = False
                st.session_state.show_meeting_data = False
                st.session_state.edit_member = None
                st.rerun()

        with col4:
            st.caption(f"Total rows: {len(edited_data)}")

    # Handle delete confirmation with error handling
    if st.session_state.delete_confirmation:
        member_to_delete = st.session_state.delete_confirmation

        st.markdown(f"""
                <div class="delete-confirmation">
                    <h4>⚠️ Confirm Deletion</h4>
                    <p>Are you sure you want to delete <strong>{member_to_delete.get('name', 'Unknown Member')}</strong>?</p>
                    <p><em>This action cannot be undone.</em></p>
                </div>
                """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("❌ Yes, Delete", type="primary", key="confirm_delete"):
                with st.spinner("Deleting member..."):
                    result = delete_member(member_to_delete['id'])
                    if result:
                        st.success(f"✅ Member '{member_to_delete['name']}' deleted successfully!")
                        st.session_state.delete_confirmation = None
                        time.sleep(1)
                        st.rerun()

        with col2:
            if st.button("↩️ Cancel", key="cancel_delete"):
                st.session_state.delete_confirmation = None
                st.rerun()
    try:
        if st.session_state.get("show_calendar"):
            show_calendar_widget()
        elif st.session_state.get("show_add_form"):
            show_add_member_form()
        elif st.session_state.get("show_view_records"):
            show_view_records()
        elif st.session_state.get("show_meeting_data"):
            show_meeting_data()
        elif st.session_state.get("edit_member"):
            show_edit_member_form()
        elif st.session_state.get("show_personal_data"):
            personal_data_module.run_as_module()

    except Exception as e:
        st.error("❌ An error occurred in the main application")
        show_error_details(str(e))

        # Recovery options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Page", key="error_refresh"):
                st.rerun()
        with col2:
            if st.button("🏠 Return to Dashboard", key="error_home"):
                st.session_state.show_add_form = False
                st.session_state.edit_member = None
                st.session_state.show_view_records = False
                st.session_state.show_meeting_data = False  # NEW
                st.rerun()


def show_dashboard_with_sidebar():
    """Combined function that shows both sidebar and dashboard after login"""
    # Show sidebar first
    show_sidebar()

    # Then show dashboard content
    show_dashboard()


@handle_error
def get_member_names_from_personal_details():
    """Fetch all member names from bni_member_personal_details table"""
    try:
        response = supabase.table('bni_member_personal_details').select('name').execute()

        if response.data:
            # Extract names and remove duplicates
            names = sorted(list(set([record['name'] for record in response.data if record.get('name')])))
            return names
        else:
            return []
    except Exception as e:
        logger.error(f"Failed to fetch member names: {e}")
        st.warning("Could not load member names from database")
        return []


def show_add_member_form():
    st.header("➕ Add Members Meeting Attendance Data")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("⬅️ Back to Dashboard", key="back_from_add_member_temp_999"):
            st.session_state.show_add_form = False
            st.rerun()

    # Initialize session state for selected date if not exists
    if 'add_member_selected_date' not in st.session_state:
        st.session_state.add_member_selected_date = datetime.now().date()

    # Date selection section at the top
    st.subheader("Select Date for Adding Record")
    st.info("Choose the date for which you want to add attendance data")

    col_date1, col_date2, col_date3 = st.columns(3)

    with col_date1:
        selected_year = st.selectbox(
            "Year",
            range(datetime.now().year - 2, datetime.now().year + 2),
            index=2,  # Current year
            key="add_member_year"
        )

    with col_date2:
        selected_month = st.selectbox(
            "Month",
            range(1, 13),
            format_func=lambda x: calendar.month_name[x],
            index=datetime.now().month - 1,  # Current month
            key="add_member_month"
        )

    with col_date3:
        # Get number of days in selected month
        days_in_month = calendar.monthrange(selected_year, selected_month)[1]
        selected_day = st.selectbox(
            "Day",
            range(1, days_in_month + 1),
            index=min(datetime.now().day - 1, days_in_month - 1),  # Current day or last day of month
            key="add_member_day"
        )

    # Construct the selected date
    selected_date = date(selected_year, selected_month, selected_day)

    # Display selected date prominently
    st.success(f"Adding data for: **{selected_date.strftime('%A, %B %d, %Y')}**")

    # NOTE: Image processing section has been moved to sidebar
    # Users can now upload/capture images from the sidebar at any time

    # Fetch member names from database
    member_names_list = get_member_names_from_personal_details()

    # Member details form
    with st.form("add_member_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Selectbox with all member names
            if member_names_list:
                name = st.selectbox(
                    "Full Name*",
                    options=[""] + member_names_list,  # Empty option at start for validation
                    help="Select a member name from the list"
                )
            else:
                st.warning("No members found in personal details table. Please add members first.")
                name = ""

            toa = st.text_input("Time of Arrival (TOA)", placeholder="e.g., 10:30 AM")

        with col2:
            mode = st.text_input("Mode", placeholder="e.g., Cash, Online, Cheque")
            payment = st.number_input(
                "Payment Amount*",
                min_value=0,
                step=100,
                help="Enter the payment amount"
            )

        submitted = st.form_submit_button("Add Attendance Details", use_container_width=True)

        if submitted:
            # Validate inputs
            if not name or name.strip() == "":
                st.error("Please select a member name")
            elif payment < 0:
                st.error("Payment amount cannot be negative")
            else:
                with st.spinner("Adding attendance details..."):
                    # Combine selected date with current time for timestamp
                    selected_datetime = datetime.combine(selected_date, datetime.now().time())

                    # Create member data with the selected date
                    member_data = {
                        'name': name.strip(),
                        'toa': toa.strip() if toa else '',
                        'payment': int(payment),
                        'mode': mode.strip(),
                        'created_at': selected_datetime.isoformat()
                    }

                    try:
                        # Insert into database
                        response = supabase.table('image_data_extraction').insert(member_data).execute()

                        if response.data:
                            st.success(
                                f"✅ Member '{name}' added successfully for {selected_date.strftime('%Y-%m-%d')}!")
                            time.sleep(1)
                            st.session_state.show_add_form = False
                            st.rerun()
                        else:
                            st.error("Failed to add member. Please try again.")
                    except Exception as e:
                        st.error(f"Database error: {str(e)}")

def show_edit_member_form():
    member_data = st.session_state.edit_member
    st.header(f"✏️ Edit Member: {member_data['name']}")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("⬅️ Back to Dashboard", key="back_from_edit_member_form_2"):
            st.session_state.edit_member = None
            st.rerun()

    try:
        st.info(
            f"**Member ID:** {member_data.get('id', 'Unknown')} | **Created:** {member_data.get('created_at', 'Unknown')}")
    except Exception as e:
        st.warning("Could not display member metadata")

    with st.form("edit_member_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name*", value=member_data.get('name', ''))

            # Safely get existing payment from member_data, default to 0
            raw_payment = member_data.get("payment", 0)

            # Convert to float for UI display
            try:
                if raw_payment is None or (isinstance(raw_payment, float) and math.isnan(raw_payment)):
                    payment_value = 0.0
                else:
                    payment_value = float(raw_payment)
            except (ValueError, TypeError):
                payment_value = 0.0

            # Show number input in UI
            payment_input = st.number_input(
                "Payment Amount*",
                min_value=0.0,
                step=100.0,
                value=payment_value,
                format="%.2f"
            )

            toa = st.text_input("Terms of Visit (TOA)", value=member_data.get('toa', ''))

        with col2:
            mode = st.text_input("Mode", value=member_data.get('mode', ''))

        submitted = st.form_submit_button("Update Member", use_container_width=True)

        if submitted:
            # Convert payment to integer for database storage
            try:
                payment = int(round(payment_input))  # Round to nearest integer to handle floating point precision
            except (ValueError, TypeError):
                st.error("Invalid payment amount. Please enter a valid number.")
                st.stop()

            with st.spinner("Updating member..."):
                member_id = member_data.get('id')
                result = update_member(member_id, name, toa, payment, mode)

                if result:
                    st.success(f"✅ Member '{name}' updated successfully!")
                    time.sleep(1)
                    st.session_state.edit_member = None
                    st.rerun()
                else:
                    st.error("Failed to update member. Please try again.")


def show_members_list():
    # Get members data with error handling
    try:
        if st.session_state.search_query:
            df = search_members(st.session_state.search_query)
            if df is not None:
                st.subheader(f"🔍 Search Results for: '{st.session_state.search_query}'")
            else:
                df = pd.DataFrame()
        else:
            df = get_all_members()
            if df is not None:
                st.subheader("👥 All Members")
            else:
                df = pd.DataFrame()

    except Exception as e:
        st.error("Failed to load members data")
        show_error_details(str(e))
        return

    # Show statistics
    stats = get_stats(df)

    col1, col2, col3 = st.columns(3)

    # Show members table with error handling
    if df is not None and not df.empty:
        try:
            # Create display dataframe with error handling
            display_df = df.copy()

            # Safe payment formatting
            if 'payment' in display_df.columns:
                display_df['payment'] = display_df['payment'].apply(
                    lambda x: f"₹{float(x):,.2f}" if pd.notnull(x) else "₹0.00"
                )

            # Safe date formatting
            if 'created_at' in display_df.columns:
                display_df['created_at'] = pd.to_datetime(
                    display_df['created_at'], errors='coerce'
                ).dt.strftime('%Y-%m-%d')

            # Show table headers
            st.markdown("---")
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1, 1, 1.5, 1])

            with col1:
                st.markdown("**Name**")
            with col2:
                st.markdown("**TOA**")
            with col3:
                st.markdown("**Payment**")
            with col4:
                st.markdown("**Mode**")

            st.markdown("---")

            # Show table with error boundaries for each row
            for idx, row in df.iterrows():
                try:
                    with st.container():
                        col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1, 1, 1.5, 1])

                        with col1:
                            st.write(f"**{row.get('name', 'Unknown')}**")
                            st.caption(f"ID: {row.get('id', 'Unknown')}")

                        with col2:
                            st.write(row.get('toa', 'Not specified'))

                        with col3:
                            payment_val = row.get('payment', 0)
                            try:
                                st.write(f"₹{float(payment_val):,.2f}")
                            except (ValueError, TypeError):
                                st.write("₹0.00")

                        with col4:
                            mode = row.get('mode', 'Unknown')

                        with col6:
                            col_edit, col_delete = st.columns(2)

                            with col_edit:
                                if st.button("✏️", key=f"edit_{row['id']}", help="Edit member"):
                                    st.session_state.edit_member = row.to_dict()
                                    st.rerun()

                            with col_delete:
                                if st.button("🗑️", key=f"delete_{row['id']}", help="Delete member"):
                                    st.session_state.delete_confirmation = row.to_dict()
                                    st.rerun()

                        st.divider()

                except Exception as e:
                    st.error(f"Error displaying member row: {str(e)}")
                    logger.error(f"Row display error for member {row.get('id', 'unknown')}: {e}")
                    continue

        except Exception as e:
            st.error("Error displaying members table")
            show_error_details(str(e))

    else:
        if st.session_state.search_query:
            st.info(f"No members found matching '{st.session_state.search_query}'")
            if st.button("🔄 Try Different Search", use_container_width=True, key="15"):
                st.session_state.search_query = ""
                st.rerun()
        else:
            st.info("No members found. Add some members to get started!")
            if st.button("➕ Add First Member", key="16"):
                st.session_state.show_add_form = True
                st.rerun()


def main():
    """Main application entry point - works as module or standalone"""
    try:
        # Show back button if running as module
        if 'selected_app' in st.session_state and st.session_state.selected_app == 'member':
            show_back_to_main_button()

        # Initialize session state
        init_session_state()

        # Global error recovery check
        if st.session_state.get('error_count', 0) > 5:
            st.error("🚨 Multiple errors detected. Resetting application state...")
            if st.button("🔄 Reset Application", key="reset_app_member"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            return

        # Check if running as module (from main app)
        if 'selected_app' in st.session_state and st.session_state.selected_app == 'member':
            # Skip login, go straight to dashboard
            show_dashboard_with_sidebar()
        else:
            # Running standalone - check login
            if not st.session_state.get('logged_in', False):
                show_login()
            else:
                show_dashboard_with_sidebar()

    except Exception as e:
        st.error("🚨 Critical application error occurred")
        show_error_details(str(e))
        increment_error_count()

        # Emergency recovery
        st.markdown("""
        ### 🆘 Emergency Recovery Options
        If the application is not working properly, try these steps:
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Page", key="refresh_member"):
                st.rerun()

        with col2:
            if st.button("🧹 Clear Session", key="clear_session_member"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        with col3:
            if st.button("🏠 Back to Main", key="error_back_member"):
                st.session_state.selected_app = None
                st.session_state.logged_in = True
                st.rerun()


# 4. FIND the if __name__ == "__main__" block at the bottom and REPLACE it with:
if __name__ == "__main__":
    # Only configure page when running standalone
    st.set_page_config(
        page_title="BNI Brilliance Member Management",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()