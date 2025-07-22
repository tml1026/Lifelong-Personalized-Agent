import os
import pickle
import json
from typing import Optional, Dict, Any
from .agent import Chatbot

class PersonaManager:
    def __init__(self, persona_db_path:str):
        self.persona_db_path = persona_db_path
        self.persona_store = self._load_persona_db()
        if self.persona_store:
            print(self.persona_store.keys())

    # def _load_persona_db(self):
    #     if os.path.exists(self.persona_db_path):
    #         if self.persona_db_path.endswith('pkl'):
    #             with open(self.persona_db_path, 'rb') as db_file:
    #                 return pickle.load(db_file)
    #         elif self.persona_db_path.endswith('json'):
    #             with open(self.persona_db_path, 'r') as db_file:
    #                 return json.load(db_file)
    #     return {}
    
    def _load_persona_db(self):
        if os.path.exists(self.persona_db_path):
            if self.persona_db_path.endswith('.pkl'):
                with open(self.persona_db_path, 'rb') as db_file:
                    try:
                        return pickle.load(db_file)
                    except (pickle.UnpicklingError, EOFError):
                       
                        return {}
            elif self.persona_db_path.endswith('.json'):
                with open(self.persona_db_path, 'r', encoding='utf-8') as db_file:
                    try:
                        return json.load(db_file)
                    except json.JSONDecodeError:
                        
                        return {}
        else:
            os.makedirs(os.path.dirname(self.persona_db_path), exist_ok=True)
        return {}

    # def _save_persona_db(self):
    #     with open(self.persona_db_path, 'wb') as db_file:
    #         pickle.dump(self.persona_store, db_file)

    def _save_persona_db(self):
        if self.persona_db_path.endswith('.pkl'):
            with open(self.persona_db_path, 'wb') as db_file:
                pickle.dump(self.persona_store, db_file)
        elif self.persona_db_path.endswith('.json'):
            with open(self.persona_db_path, 'w', encoding='utf-8') as db_file:
                json.dump(self.persona_store, db_file, ensure_ascii=False)

    def load_persona(self, user_id: str) -> Optional[str]:
        self.persona_store = self._load_persona_db()
        return self.persona_store.get(user_id, None)

    def set_persona(self, user_id: str, persona: dict):
        self.persona_store[user_id] = persona
        self._save_persona_db()

    def update_field_content(self, user_id: str, field: str, new_content: Any):
        persona = self.load_persona(user_id)
        if persona and field in persona:
            if isinstance(persona[field], dict):
                for sub_field in persona[field]:
                    if new_content.get(sub_field):
                        persona[field][sub_field] = new_content[sub_field]
            else:
                persona[field] = new_content
            self.set_persona(user_id, persona)
        else:
            raise ValueError(f"Field '{field}' does not exist in the persona for user_id '{user_id}'.")

    def update_fields(self, user_id: str, fields_to_update: Dict[str, Any]):
        persona = self.load_persona(user_id)
        if not persona:
            raise ValueError(f"No persona found for user_id '{user_id}'.")

        for field, new_content in fields_to_update.items():
            self.update_field_content(user_id, field, new_content)
    

    def print_persona(self,user_id) -> str:
        persona = self.load_persona(user_id)
        if not persona:
            return "No persona found for the current user."
        summary ="--"*10
        summary += f"\nUser_{user_id} Persona Summary:\n"
        for key, value in persona.items():
            if key == "scenes":
                continue
            if isinstance(value, dict):
                summary += f"{key}:\n"
                for sub_key, sub_value in value.items():
                    summary += f"  {sub_key}: {sub_value}\n"
            else:
                summary += f"{key}: {value}\n"
        return summary+"\n---------------------"

    def print_persona_db(self):
        print("[Persona DB 全部内容]:", self.persona_store)

    