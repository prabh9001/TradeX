
import os
from flask import Flask
from app import db, User, Portfolio, app

def inspect_database():
    print("\n--- TradeX Database Inspector ---")
    
    # Check if DB file exists
    db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found. Initializing...")
        with app.app_context():
            db.create_all()
            print("Database created successfully.")
    
    with app.app_context():
        # List tables
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"Tables found: {', '.join(tables)}")
        
        # Show User table structure
        print("\n[Table: User]")
        for column in inspector.get_columns('user'):
            print(f"- {column['name']}: {column['type']}")
            
        # Show Portfolio table structure
        print("\n[Table: Portfolio]")
        for column in inspector.get_columns('portfolio'):
            print(f"- {column['name']}: {column['type']}")
            
        # Sample counts
        user_count = User.query.count()
        portfolio_count = Portfolio.query.count()
        print(f"\nStats: {user_count} users, {portfolio_count} portfolio items.")

if __name__ == "__main__":
    inspect_database()
