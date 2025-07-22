from .user_simulator import User
from .personalized_agent import PersonalizedChatbot
class ConversationManager:
    def __init__(self):
        self.conversation_id = None
    
    def start_new_conversation(self, user: User, personalized_chatbot: PersonalizedChatbot):
        
