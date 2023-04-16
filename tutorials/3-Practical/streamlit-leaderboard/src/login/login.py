from typing import Optional

import streamlit as st

from src.login.username_password_manager import UsernamePasswordManagerArgon2
from src.common.utils import remove_illegal_filename_characters, is_legal_filename

class LoginState:
    def __init__(self, username: str, is_logged_in: bool):
        self.username = username
        self.is_logged_in = is_logged_in

class Login:
    def __init__(self, password_manager: UsernamePasswordManagerArgon2, max_num_users: Optional[int] = None):
        self.password_manager = password_manager
        self.max_num_users = max_num_users
        self.session_state = LoginState(username='', is_logged_in=False)

        # placeholders
        self.first_login_checkbox_placeholder = None
        self.username_placeholder = None
        self.pwd_placeholder = None
        self.pwd2_placeholder = None
        self.login_button_placeholder = None
        self.signout_button_placeholder = None

    def init(self):
        self.first_login_checkbox_placeholder = st.sidebar.empty()
        self.username_placeholder = st.sidebar.empty()
        self.pwd_placeholder = st.sidebar.empty()
        self.pwd2_placeholder = st.sidebar.empty()
        self.login_button_placeholder = st.sidebar.empty()
        self.signout_button_placeholder = st.sidebar.empty()

    def get_username(self) -> str:
        return self.session_state.username

    def is_logged_in(self) -> bool:
        return self.session_state.is_logged_in

    def has_user_limit_been_reached(self) -> bool:
        if self.max_num_users is None:
            return False
        return self.max_num_users <= len(self.password_manager.get_all_usernames())

    def clear_placeholders(self):
        self.first_login_checkbox_placeholder.empty()
        self.username_placeholder.empty()
        self.pwd_placeholder.empty()
        self.pwd2_placeholder.empty()
        self.login_button_placeholder.empty()
        self.signout_button_placeholder.empty()

    def run_and_return_if_access_is_allowed(self) -> bool:
        if (not self.password_manager.is_username_taken(self.session_state.username)) or \
                (not self.session_state.is_logged_in):
            self.session_state.is_logged_in = False
            is_first_login = self.first_login_checkbox_placeholder.checkbox("This is my first login", value=False)
            if is_first_login:
                is_logged_in = self.try_signup()
            else:
                is_logged_in = self.try_login()
            if is_logged_in:
                self.clear_placeholders()
            return is_logged_in
        return True

    def try_login(self) -> bool:
        username = self.username_placeholder.text_input("Username:", value="", max_chars=30)
        pwd = self.pwd_placeholder.text_input("Password:", value="", type="password", max_chars=30)
        login_button = self.login_button_placeholder.button("Login")
        if not login_button:
            return False
        if (self.password_manager.is_username_taken(username)) and \
                (self.password_manager.verify(username, pwd)):
            self.session_state.username = username
            self.session_state.is_logged_in = True
            return True
        else:
            st.sidebar.error("The username or password you entered is incorrect")
            return False

    def _is_valid_username(self, username: str) -> bool:
        return (len(username) > 0) and is_legal_filename(username)

    def try_signup(self) -> bool:
        username = self.username_placeholder.text_input("Username:", value="", max_chars=30)
        pwd = self.pwd_placeholder.text_input("Password:", value="", type="password", max_chars=30)
        pwd2 = self.pwd2_placeholder.text_input("Retype password:", value="", type="password", max_chars=30)
        signup_button = self.login_button_placeholder.button("Sign up")
        if not signup_button:
            return False

        if self.has_user_limit_been_reached():
            st.sidebar.error("The user limit has been reached. Contact the admin for assistance.")
        elif not self._is_valid_username(username):
            st.sidebar.error('Invalid username. Must have only alphanumeric or ".-_ " characters, '
                             'without trailing or leading whitespaces.')
        elif self.password_manager.is_username_taken(username):
            st.sidebar.error("Username already exists.")
        elif pwd != pwd2:
            st.sidebar.error('Passwords do not match.')
        elif len(pwd) == 0:
            st.sidebar.error('Please choose a password')
        else:
            self.session_state.username = username
            self.password_manager.store(username, pwd)
            self.session_state.is_logged_in = True
            return True
        return False

    def has_user_signed_out(self) -> bool:
        if not self.session_state.is_logged_in:
            return False
        if self.signout_button_placeholder.button("Sign out"):
            self.session_state.username = ''
            self.session_state.is_logged_in = False
            self.clear_placeholders()
            self.run_and_return_if_access_is_allowed()
            return True
        return False
