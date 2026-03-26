"""CLI for managing users. Usage: python -m app.manage add-user <email>"""
import sys
import getpass

from app.database import init_db, SessionLocal
from app.models import User, UserPreference
from app.auth import hash_password


def add_user(email: str):
    init_db()
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"User {email} already exists.")
            return

        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match.")
            return

        user = User(email=email, password_hash=hash_password(password))
        db.add(user)
        db.flush()
        db.add(UserPreference(user_id=user.id))
        db.commit()
        print(f"User {email} created successfully.")
    finally:
        db.close()


def list_users():
    init_db()
    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            print("No users found.")
            return
        for u in users:
            print(f"  {u.id}: {u.email}")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.manage <command> [args]")
        print("Commands: add-user <email>, list-users")
        sys.exit(1)

    command = sys.argv[1]
    if command == "add-user":
        if len(sys.argv) < 3:
            print("Usage: python -m app.manage add-user <email>")
            sys.exit(1)
        add_user(sys.argv[2])
    elif command == "list-users":
        list_users()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
