"""Database interface to interact with the mongoDB instance"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from the root `.env`
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
ENV_PATH = BASE_DIR / "../.env"

# Load environment variables safely
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print("Warning: `.env` file not found! Using default values.")


class MemoryDB:
    """Long-term memory for Binge Buddy"""

    def __init__(self):
        self.host = os.getenv("MONGO_HOST", "localhost")
        self.port = int(os.getenv("MONGO_PORT", "27017"))
        self.db_name = os.getenv("MONGO_DB_NAME", "binge_buddy_db")
        self.username = os.getenv("MONGO_USER", "root")
        self.password = os.getenv("MONGO_PASS", "rootpass")

        # Connection URI with authentication
        self.uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/"

        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            sys.exit()

    def get_collection(self, collection_name):
        """Get a reference to a collection."""
        return self.db[collection_name]

    def insert_one(self, collection_name, data):
        """Insert a single document."""
        return self.get_collection(collection_name).insert_one(data)

    def find_one(self, collection_name, query):
        """Find a single document."""
        return self.get_collection(collection_name).find_one(query)

    def update_one(self, collection_name, query, update_data):
        """Update a single document."""
        return self.get_collection(collection_name).update_one(
            query, {"$set": update_data}
        )

    def delete_one(self, collection_name, query):
        """Delete a single document."""
        return self.get_collection(collection_name).delete_one(query)

    def close(self):
        """Close the database connection."""
        self.client.close()
