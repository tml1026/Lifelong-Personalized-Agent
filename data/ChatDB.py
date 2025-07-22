import os
import pickle
from typing import Optional
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

class SessionHistoryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.store = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as db_file:
                return pickle.load(db_file)
        return {}

    def _save_db(self):
        with open(self.db_path, 'wb') as db_file:
            pickle.dump(self.store, db_file)

    def get_conversation_history(self, user_id: str, conversation_id: Optional[int] = None) -> BaseChatMessageHistory:
        # Initialize user history if not present
        if user_id not in [key[0] for key in self.store.keys()]:
            self.store[(user_id, 1)] = InMemoryChatMessageHistory()  # Initialize with a new conversation ID (1)
            self._save_db()

        if conversation_id is None:
            # If conversation_id is not specified, get the latest conversation for the user
            user_conversations = [key for key in self.store.keys() if key[0] == user_id]
            conversation_id = max([key[1] for key in user_conversations])

        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = InMemoryChatMessageHistory()

        return self.store[(user_id, conversation_id)]

    def get_session_history(self, user_id: str) -> BaseChatMessageHistory:
        # Combine all conversation histories for the user_id into one
        combined_history = InMemoryChatMessageHistory()
        user_conversations = sorted([key[1] for key in self.store.keys() if key[0] == user_id])

        for conversation_id in user_conversations:
            conversation_history = self.get_conversation_history(user_id, conversation_id)
            combined_history.messages.extend(conversation_history.messages)

        return combined_history

    def add_message(self, user_id: str, conversation_id: int, message, is_user: bool):
        history = self.get_conversation_history(user_id, conversation_id)
        if is_user:
            history.add_user_message(message)
        else:
            history.add_ai_message(message)
        self._save_db()

    def start_new_conversation(self, user_id: str) -> int:
        # Initialize user history if not present
        if user_id not in [key[0] for key in self.store.keys()]:
            self.store[(user_id, 1)] = InMemoryChatMessageHistory()  # Initialize with a new conversation ID (1)
            self._save_db()
            return 1

        user_conversations = [key for key in self.store.keys() if key[0] == user_id]
        conversation_id = max([key[1] for key in user_conversations]) + 1
        self.store[(user_id, conversation_id)] = InMemoryChatMessageHistory()
        self._save_db()
        return conversation_id

    def print_conversation_history(self, user_id: str, conversation_id: int) -> str:
        history = self.get_conversation_history(user_id, conversation_id)
        return "\n".join(
            [f"用户: {msg.content}" if isinstance(msg,HumanMessage) else f"AI助手: {msg.content}" for msg in history.messages]
        )
    
    def formulate_conversation_history(self, user_id: str, conversation_id: int) -> str:
        history = self.get_conversation_history(user_id, conversation_id)
        return [{"User": f"{msg.content}"} if isinstance(msg,HumanMessage) else {"AI": f"{msg.content}"} for msg in history.messages]

    def get_latest_conversation(self,user_id:str) -> BaseChatMessageHistory:
    # Find all conversation IDs for the given user_id
        user_conversations = sorted([key[1] for key in self.store.keys() if key[0] == user_id])
        
        # Check if there are any conversations for the user
        if not user_conversations:
            print("no conversation history found")
            return InMemoryChatMessageHistory()  # Return an empty history if no conversations exist
        
        # Get the latest conversation ID
        latest_conversation_id = user_conversations[-1]
        
        # Retrieve and return the conversation history for the latest conversation
        return self.get_conversation_history(user_id, latest_conversation_id)
        
if __name__ == "__main__":
    db_path = "chat_history.db"
    session_db = SessionHistoryDB(db_path=db_path)
    session_db.store.keys()
    chat_history = session_db.get_latest_conversation('anonymous_1')
    print(chat_history)