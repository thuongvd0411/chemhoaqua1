import json
import os

DATA_FILE = "users_data.json"

class DataManager:
    def __init__(self):
        self.users = self.load_data()

    def load_data(self):
        if not os.path.exists(DATA_FILE):
            return {}
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def save_data(self):
        try:
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self.users, f, ensure_ascii=False, indent=4)
        except Exception as e:
            # On Streamlit Cloud or read-only systems, just ignore save
            print(f"Warning: Could not save data: {e}")

    def login(self, username):
        """Returns user dict if exists, else None"""
        return self.users.get(username)

    def register(self, username, display_name):
        if username in self.users:
            return False, "Tên đăng nhập đã tồn tại!"
        
        self.users[username] = {
            "display_name": display_name,
            "stars": 0,
            "gender": "Khác",
            "selected_rank": None,
            "history": []
        }
        self.save_data()
        return True, self.users[username]

    def add_stars(self, username, amount):
        if username in self.users:
            self.users[username]["stars"] += amount
            
            # Add to history
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            if "history" not in self.users[username]:
                self.users[username]["history"] = []
            
            self.users[username]["history"].append({
                "date": timestamp,
                "score": amount
            })
            
            self.save_data()
            return True
        return False

    def update_profile(self, username, display_name, gender, selected_rank):
        if username in self.users:
            self.users[username]["display_name"] = display_name
            self.users[username]["gender"] = gender
            self.users[username]["selected_rank"] = selected_rank
            self.save_data()
            return True
        return False

    def get_leaderboard(self):
        # Return list of user dicts sorted by stars desc
        items = []
        for username, u in self.users.items():
            user_copy = u.copy()
            user_copy['username'] = username
            items.append(user_copy)
        return sorted(items, key=lambda x: x["stars"], reverse=True)
