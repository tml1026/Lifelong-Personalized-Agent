from typing import Dict, Any, List,Optional
from pydantic import Field
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from data import SessionHistoryDB
import torch

class Chatbot:
    def __init__(self, user_id:Optional[str], db_path: Optional[str],openai_api_key: str,openai_api_base:str, model_name: str = "gpt-4o", temperature: float = 0.9):
        self.model = ChatOpenAI(openai_api_key=openai_api_key, openai_api_base=openai_api_base, model=model_name, temperature=temperature)
        self.user_id = user_id if user_id else None
        self.session_db = SessionHistoryDB(db_path) if db_path else None
        self.sentence_model = None
        print("[INFO] SentenceTransformer已跳过加载（测试模式）")

    def load_session_db(self,db_path):
        try:
            self.session_db = SessionHistoryDB(db_path)
        except:
            raise "Not a valid db_path, please check again"
    
    def set_user(self,user_id: str):
        self.user_id = user_id


    def retrieve_sessions(self, user_id: str, user_query: str, top_k: int = 1):

        if self.sentence_model is None:
            print("⚠️ SentenceTransformer未加载，跳过会话检索功能")
            return []

        user_conversations = sorted([key[1] for key in self.session_db.store.keys() if key[0] == user_id])
        all_conversations = [] 
        message_to_conversation_map = [] 

        for conversation_id in user_conversations:
            conversation_history = self.session_db.get_conversation_history(user_id, conversation_id)
            conversation_text = []  
            for message in conversation_history.messages:
                if isinstance(message, HumanMessage):
                    role = "User"
                elif isinstance(message, AIMessage):
                    role = "AI"
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")
                
                conversation_text.append(f"{role}: {message.content}")
                message_to_conversation_map.append((message.content, conversation_id))
            
            all_conversations.append("\n".join(conversation_text))

        if not message_to_conversation_map:
            return []

        message_texts = [msg[0] for msg in message_to_conversation_map]

        query_embedding = self.sentence_model.encode(user_query, convert_to_tensor=True)
        message_embeddings = self.sentence_model.encode(message_texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, message_embeddings)

        top_results = torch.topk(similarities, k=min(top_k, len(message_texts)))
        relevant_conversation_ids = set(
            message_to_conversation_map[idx][1] for idx in top_results.indices[0]
        )

        relevant_sessions = [
            {
                "conversation_id": conversation_id,
                "chat_history": all_conversations[user_conversations.index(conversation_id)],
                "score": similarities[0][list(message_texts).index(msg)].item(),
            }
            for conversation_id in relevant_conversation_ids
            for msg in message_texts if message_to_conversation_map[message_texts.index(msg)][1] == conversation_id
        ]

        relevant_sessions = sorted(relevant_sessions, key=lambda x: x["score"], reverse=True)
        return relevant_sessions[:top_k]

    def start_new_conversation(self):
        if not self.user_id:
            raise ValueError("User ID is not set. Please set the user using set_user().")
        self.conversation_id = self.session_db.start_new_conversation(self.user_id)

    def add_message(self, message: str, is_user: bool = True):
        if not self.conversation_id:
            raise ValueError("Conversation ID is not set. Please start a new conversation.")
        self.session_db.add_message(self.user_id, self.conversation_id, message, is_user)
    
    def general_prompt_manager(self,  query: str):
        system_prompt = "你是一个在终端上部署的AI助手，除了像ChatGPT一样的多轮对话之外，你还可以访问、操作终端上的应用，以及进行联网搜索。\n请你尽你所能帮助用户！"
        system_message = SystemMessage(content=system_prompt)
        history_messages = self.session_db.get_session_history(self.user_id, self.current_user_id).messages
        messages = [system_message] + history_messages
        messages.append(HumanMessage(content=query))
        self.add_message(query, is_user=True) 

        return messages

    def get_response(self, messages: List, if_add_message:bool=True) -> str:
        if not self.conversation_id and if_add_message:
            raise ValueError("Conversation ID is not set. Please start a new conversation using start_new_conversation().")

        # print(messages)
        response = self.model.invoke(messages)

        if if_add_message:
            self.add_message(response.content, is_user=False)
        return response.content
        # return response
    
    def get_session_history(self) -> Dict[str, List[Dict[str, str]]]:
        if not self.user_id:
            raise ValueError("User ID is not set. Please set the user using set_user().")
        
        all_histories = {}
        user_conversations = [key for key in self.session_db.store.keys() if key[0] == self.user_id]
        
        for conversation_id in user_conversations:
            # conversation_history = self.session_db.get_session_history(self.user_id, conversation_id[1]).messages
            conversation_history = self.session_db.get_session_history(self.user_id).messages
            formatted_history = [
                {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content} for msg in conversation_history
            ]
            all_histories[str(conversation_id[1])] = formatted_history
        
        return all_histories

    def get_conversation_history(self,conversation_id:Optional[int]=None) -> str:
        if not self.conversation_id and not conversation_id:
            raise ValueError("Conversation ID is not set. Please indict an exact conversation number or start a new conversation.")
        conversation_id = conversation_id or self.conversation_id
        return self.session_db.get_conversation_history(self.user_id, conversation_id)

    def formulate_conversation_history(self,conversation_id:Optional[int]=None):
        if not self.conversation_id and not conversation_id:
            raise ValueError("Conversation ID is not set. Please indict an exact conversation number or start a new conversation.")
        conversation_id = conversation_id or self.conversation_id
        return self.session_db.formulate_conversation_history(self.user_id,conversation_id)


if __name__ == "__main__":
    chatbot = Chatbot(user_id=None, db_path="chat_history.db", openai_api_key="your-api-key")
    chatbot.set_user(123)
    chatbot.start_new_conversation()
    chatbot.get_response(persona="persona details", scene="scene details", api_docs="api docs", user_input="Hello!")
    print(chatbot.get_conversation_history())
    print(chatbot.get_session_history())