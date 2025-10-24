import streamlit as st
import hashlib
from datetime import datetime
import logging
from supabase import create_client, Client

# Configure Streamlit page - MUST be first
st.set_page_config(
    page_title="BNI Brilliance Management System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]


# Initialize Supabase client
@st.cache_resource
def init_supabase():
    try:
        client = create_client(supabase_url, supabase_key)
        client.table('admins').select('id').limit(1).execute()
        return client
    except Exception as e:
        logger.error(f"Supabase initialization failed: {e}")
        st.error("❌ Database Connection Failed")
        st.stop()


supabase: Client = init_supabase()

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }

    .big-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .big-button:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }

    .member-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .visitor-button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    .login-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate_user(username: str, password: str) -> bool:
    """Authenticate admin user"""
    try:
        password_hash = hash_password(password)
        response = supabase.table('admins').select('*').eq('username', username.strip()).execute()

        if not response.data:
            return False

        stored_user = response.data[0]
        stored_hash = stored_user.get('password_hash', '')

        if stored_hash == password_hash:
            logger.info(f"Successful login for user: {username}")
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False


def ensure_admin_exists() -> bool:
    """Create default admin if not exists"""
    try:
        admin_response = supabase.table('admins').select('*').eq('username', 'admin').execute()

        if len(admin_response.data) == 0:
            default_password = 'admin123'
            admin_data = {
                'username': 'admin',
                'password_hash': hash_password(default_password),
                'created_at': datetime.now().isoformat()
            }

            result = supabase.table('admins').insert(admin_data).execute()

            if result.data:
                logger.info("Default admin created successfully")
                return True
        return False

    except Exception as e:
        logger.error(f"Admin creation failed: {e}")
        return False


def reset_admin_password() -> bool:
    """Reset admin password to default"""
    try:
        default_password = 'admin123'
        new_hash = hash_password(default_password)

        supabase.table('admins').delete().eq('username', 'admin').execute()

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
        return False

    except Exception as e:
        logger.error(f"Password reset failed: {e}")
        return False


def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'selected_app' not in st.session_state:
        st.session_state.selected_app = None


def show_login():
    """Display login page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🏢 BNI Brilliance Management System</h1>
        <p style="font-size: 1.2rem;">Enter your credentials to access the system</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            #st.markdown('<div class="login-container">', unsafe_allow_html=True)

            username = st.text_input("Username", placeholder="Enter admin username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("🔐 Login", use_container_width=True)

            if submitted:
                with st.spinner("Authenticating..."):
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")

            st.markdown('</div>', unsafe_allow_html=True)

        st.info("""
        **Default Credentials:**  
        Username: `admin`  
        Password: `admin123`
        """)

        st.divider()
        st.subheader("🆘 Having trouble logging in?")

        col_reset1, col_reset2 = st.columns(2)

        with col_reset1:
            if st.button("🔄 Reset Admin Password", help="Resets admin password to default"):
                reset_admin_password()

        with col_reset2:
            if st.button("🔧 Create Admin Account", help="Creates admin account if missing"):
                if ensure_admin_exists():
                    st.success("✅ Admin account created!")
                    st.info("Username: admin | Password: admin123")


def show_dashboard():
    """Display main dashboard with two app options"""

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1>🏢 BNI Brilliance Management System</h1>
        
    </div>
    """, unsafe_allow_html=True)

    # Logout button at top right
    col1, col2, col3 = st.columns([4, 1, 1])
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        # Two big buttons for app selection
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("Member Attendance", key="member_btn", type="primary"):
            st.session_state.selected_app = "member"
            st.rerun()

        if st.button("Visitor Attendance", key="visitor_btn", type="primary"):
            st.session_state.selected_app = "visitor"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def run_member_app():
    """Run member app with error handling"""
    try:
        # Import at the time of use
        import python_member_app

        # Check if main function exists
        if hasattr(python_member_app, 'main'):
            python_member_app.main()
        else:
            # Fallback: try to run the app directly
            if hasattr(python_member_app, 'show_dashboard_with_sidebar'):
                # Add back button
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button("⬅️ Back to Main", key="back_to_main_member"):
                        st.session_state.selected_app = None
                        st.rerun()

                # Initialize and run
                python_member_app.init_session_state()
                python_member_app.show_dashboard_with_sidebar()
            else:
                st.error("❌ Member app module not properly configured")
                st.info("Please ensure python_member_app.py has a main() function")
                if st.button("🏠 Back to Dashboard"):
                    st.session_state.selected_app = None
                    st.rerun()
    except Exception as e:
        st.error(f"❌ Error loading member app: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        if st.button("🏠 Back to Dashboard"):
            st.session_state.selected_app = None
            st.rerun()


def run_visitor_app():
    """Run visitor app with error handling"""
    try:
        # Import at the time of use
        import BNI_Visitors_Records

        # Check if main function exists
        if hasattr(BNI_Visitors_Records, 'main'):
            BNI_Visitors_Records.main()
        else:
            # Fallback: try to run the app directly
            if hasattr(BNI_Visitors_Records, 'show_dashboard_with_sidebar'):
                # Add back button
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button("⬅️ Back to Main", key="back_to_main_visitor"):
                        st.session_state.selected_app = None
                        st.rerun()

                # Initialize and run
                BNI_Visitors_Records.init_session_state()
                BNI_Visitors_Records.show_dashboard_with_sidebar()
            else:
                st.error("❌ Visitor app module not properly configured")
                st.info("Please ensure BNI_Visitors_Records.py has a main() function")
                if st.button("🏠 Back to Dashboard"):
                    st.session_state.selected_app = None
                    st.rerun()
    except Exception as e:
        st.error(f"❌ Error loading visitor app: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        if st.button("🏠 Back to Dashboard"):
            st.session_state.selected_app = None
            st.rerun()


def main():
    """Main application entry point"""
    init_session_state()

    if not st.session_state.logged_in:
        show_login()
    elif st.session_state.selected_app == "member":
        run_member_app()
    elif st.session_state.selected_app == "visitor":
        run_visitor_app()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()