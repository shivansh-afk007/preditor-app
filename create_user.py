from werkzeug.security import generate_password_hash
import json
import os

def create_user(username, password, role):
    # Load existing users
    users_file = 'users.json'
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            users = json.load(f)
    else:
        users = {}
    
    # Generate new user ID
    user_id = str(len(users) + 1)
    
    # Create new user with hashed password
    users[user_id] = {
        'username': username,
        'password': generate_password_hash(password),
        'role': role
    }
    
    # Save updated users
    with open(users_file, 'w') as f:
        json.dump(users, f, indent=4)
    
    print(f"User {username} created successfully!")

if __name__ == '__main__':
    username = input("Enter username: ")
    password = input("Enter password: ")
    role = input("Enter role (consumer/lender): ")
    create_user(username, password, role) 