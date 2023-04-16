import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Set, Dict
import argon2


class UsernamePasswordManagerArgon2:
    def __init__(self, passwords_db_filepath: Path, **argon2_kwargs):
        self.passwords_db_filepath = str(passwords_db_filepath)
        self._create_db()
        self.argon2_hasher = argon2.PasswordHasher(**argon2_kwargs)

    def _create_db(self):
        with sqlite3.connect(self.passwords_db_filepath) as con:
            with closing(con.cursor()) as cursor:
                cursor.execute("CREATE TABLE IF NOT EXISTS passwords (user TEXT, hash TEXT)")

    def _load_user2hash(self) -> Dict[str, str]:
        with sqlite3.connect(self.passwords_db_filepath) as con:
            with closing(con.cursor()) as cursor:
                user_hash_tuples = cursor.execute("SELECT user, hash FROM passwords").fetchall()
        return {user: hash for user, hash in user_hash_tuples}

    def store(self, username: str, password: str):
        with sqlite3.connect(self.passwords_db_filepath) as con:
            with closing(con.cursor()) as cursor:
                cursor.execute("INSERT INTO passwords VALUES (?, ?)", (username, self.argon2_hasher.hash(password)))

    def verify(self, username: str, password: str) -> bool:
        try:
            user2hash = self._load_user2hash()
            return self.argon2_hasher.verify(user2hash[username], password)
        except:
            return False

    def get_all_usernames(self) -> Set[str]:
        user2hash = self._load_user2hash()
        return set(user2hash.keys())

    def is_username_taken(self, username) -> bool:
        user2hash = self._load_user2hash()
        return username in user2hash
