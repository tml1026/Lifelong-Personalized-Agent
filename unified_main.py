import os
import json
import logging
import argparse
import time
import re
from typing import Dict, Any, Optional
from pathlib import Path

from modules import PersonalizedChatbot, PersonaManager, User, APICaller
from utils.common import parse_api_call, setup_logging, print_scene


def update_persona_from_fields(fields_str, persona):
    match = re.search(r'<fields>(.*?)</fields>', fields_str, re.DOTALL)
    if not match:
        print("未找到 <fields> 标签中的内容")
        return persona

    json_str = match.group(1)
    try:
        fields_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return persona

    updated_persona = json.loads(json.dumps(persona))

    for key, value in fields_data.items():
        if key in ["scenes", "Demographics"]:
            continue
        
        found = False
        if key in updated_persona:
            if isinstance(updated_persona[key], dict) and isinstance(value, dict):
                updated_persona[key].update(value)
            else:
                updated_persona[key] = value
            found = True
        else:
            for top_level_value in updated_persona.values():
                if isinstance(top_level_value, dict) and key in top_level_value:
                    top_level_value[key] = value
                    found = True
                    break
    
    return updated_persona


class LearningExperimentRunner:
    
    def __init__(self, persona_file: str, persona_db_file: str, update_frequency: int, openai_api_key: str, 
                 openai_api_base: str, model_name: str, log_dir: str, db_prefix: str, 
                 max_retries: int, log_level: str):
        self.persona_file = persona_file
        self.persona_db_file = persona_db_file
        self.update_frequency = update_frequency  
        
        # API配置
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.model_name = model_name
        
        self.log_dir = log_dir
        self.db_prefix = db_prefix
        self.max_retries = max_retries
        self.log_level = log_level
        
        self.filename = os.path.basename(persona_file)
        self.name = self.filename.replace('.json', '')

        self.chat_lengths = []
        self.accumulated_chat_histories = []
        
        self.k = 0  
        
        self._setup_logging()
        
    def _setup_logging(self):
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, f"{self.name}.log")
        
        logging.basicConfig(
            filename=log_file, 
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _load_persona_data(self) -> tuple:
        with open(self.persona_file, 'r', encoding='utf-8') as f:
            user_persona = json.load(f)
        if isinstance(user_persona, dict):
            if "Name" in user_persona and "Demographics" in user_persona:
                user_id = user_persona["Name"]
                user_persona = {user_id: user_persona}
            else:
                keys = list(user_persona.keys())
                user_id = keys[0] if keys else None
                if not user_id:
                    raise ValueError("No user_id found in persona file")
        else:
            raise ValueError("Invalid persona data format")
        return user_persona, user_id
    
    def _setup_components(self, user_id: str, user_persona: Dict) -> tuple:
        db_path = f"data/{self.db_prefix}/{self.name}.db"
        
        persona_manager = PersonaManager(self.persona_db_file)
        
        chatbot = PersonalizedChatbot(
            user_id=user_id,
            persona_manager=persona_manager,
            db_path=db_path,
            openai_api_key=self.openai_api_key,
            openai_api_base=self.openai_api_base,
            model_name=self.model_name
        )
        
        api_caller = APICaller(
            openai_api_key=self.openai_api_key,
            openai_api_base=self.openai_api_base,
            model_name=self.model_name
        )
        
        user = User(
            user_id=user_id,
            db_path=db_path,
            persona=user_persona,
            openai_api_key=self.openai_api_key,
            openai_api_base=self.openai_api_base,
            model_name=self.model_name
        )
        
        return chatbot, api_caller, user, persona_manager
    
    def _process_scene(self, scene: Dict, chatbot, api_caller, user, persona=None) -> Dict:
        scene_desc = scene["scene_desc"]
        expected_response = scene["expected_response"]
        api_docs = scene.get("available_apis", "")
        
        user_query = scene.get("neutralized_query", "").replace(
            '\n</neutralized_query>', ''
        ).replace('<neutralized_query>\n', '')
        scene['neutralized_query'] = user_query
        user.set_scene(scene_desc)
        
        if persona:
            filtered_persona = {k: v for k, v in persona.items() if k != "scenes"}
            chatbot.set_persona(persona=filtered_persona)
            logging.info(f"Persona: {filtered_persona}")
        
        chatbot.start_new_conversation()
        logging.info(f"Conversation ID: {chatbot.conversation_id}")
        
        satisfied = False
        retries = 0
        
        while not satisfied and retries < self.max_retries:
            retries += 1
            
            messages = chatbot.build_persona_query_prompt(
                scene=scene_desc, 
                api_docs=str(api_docs), 
                query=user_query
            )
            
            chatbot_response = chatbot.get_response(messages)
            
            time.sleep(2)
            
            api_call = parse_api_call(chatbot_response)
            if api_call:
                logging.info(f"API call detected: {api_call}")
                api_call_messages = api_caller.api_call_prompt_manager(api_call, api_docs=api_docs)
                api_response = api_caller.get_response(api_call_messages)
                
                messages = chatbot.build_prompt_with_api_response(
                    scene=scene_desc, 
                    api_docs=api_docs, 
                    api_response=api_response
                )
                chatbot_response = chatbot.get_response(messages=messages)
            
            conversation_history = chatbot.get_conversation_history()
            satisfied = user.satisfaction_check(
                expected_results=expected_response, 
                conversation_history=conversation_history
            )
            
            logging.info(f"Retry {retries}, Satisfied: {satisfied}")
            
            if satisfied or retries >= self.max_retries:
                formatted_conversation_history = chatbot.formulate_conversation_history()
                scene['conversations'] = formatted_conversation_history
                
                output_dir = f"data/{self.db_prefix}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{self.name}.json"
                
                with open(output_file, 'a', encoding='utf-8') as g:
                    g.write(json.dumps(scene, ensure_ascii=False) + '\n')
                break
            else:
                user_query = user.get_query(scene_desc, scene.get('scene_context', ''), chat_history=conversation_history)
        
        serializable_history = []
        if hasattr(conversation_history, 'messages'):
            for msg in conversation_history.messages:
                if hasattr(msg, 'content'):
                    role = "user" if "Human" in str(type(msg)) else "assistant"
                    serializable_history.append({
                        "role": role,
                        "content": msg.content
                    })
        
        return {
            "conversation_history": serializable_history,
            "satisfied": satisfied,
            "retries": retries
        }
    
    def run(self):
        try:
            persona_manager = PersonaManager(self.persona_db_file)
            user_persona, user_id = self._load_persona_data()
            persona = persona_manager.load_persona(user_id)
            if persona is None:
                logging.warning(f"PersonaManager中未找到user_id {user_id}，使用原始文件初始化")
                persona = user_persona[user_id]
                persona_manager.set_persona(user_id, persona)
                logging.info(f"已将persona数据保存到PersonaManager for user_id {user_id}")
            elif isinstance(persona, str):
                raise ValueError("PersonaManager返回了字符串而不是字典")
            print("[检查] 当前数据库 persona:", persona_manager.load_persona(user_id))
            chatbot, api_caller, user, _ = self._setup_components(user_id, user_persona[user_id])
            persona = persona_manager.load_persona(user_id)
            scenes = persona.get("scenes", [])
            
            results = []
            for idx, scene in enumerate(scenes):
                self.k += 1
                
                logging.info(f"Processing scene {self.k} for user_id {user_id}")
                print(f"-----Scene {self.k}----")
                print_scene(scene)
                
                result = self._process_scene(scene, chatbot, api_caller, user, persona)
                results.append(result)
                
                chat_history = chatbot.session_db.get_latest_conversation(chatbot.user_id)
                user_messages = []
                for msg in chat_history.messages:
                    if hasattr(msg, 'content') and "Human" in str(type(msg)):
                        user_messages.append(msg)
                
                conversation_length = len(user_messages)
                self.chat_lengths.append(conversation_length)
                self.accumulated_chat_histories.append(chat_history)
                
                if len(self.accumulated_chat_histories) > 3:
                    self.accumulated_chat_histories.pop(0)  
                                
                if (self.k % self.update_frequency == 0 or (len(scenes) - idx - 1) < 3):
                    self._update_persona(chatbot, persona_manager, user_id)
                    persona = persona_manager.load_persona(user_id)
            
            self._save_results(user_id, results)
            
            print("Chat Lengths:", self.chat_lengths)
            logging.info(f"Chat Lengths: {self.chat_lengths}")
            
            print("[检查] 运行结束后数据库 persona:", persona_manager.load_persona(user_id))
            logging.info(f"Completed processing for user {user_id}")
            
        except Exception as e:
            logging.error(f"Error processing {self.persona_file}: {str(e)}")
            raise
    
    def _save_results(self, user_id: str, results: list):
        results_dir = f"results/learning"
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = f"{results_dir}/{user_id}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def _update_persona(self, chatbot, persona_manager, user_id: str):
        """更新persona"""
        try:
            combined_chat_history = []
            for chat in self.accumulated_chat_histories:
                if len(chat.messages) > 9:
                    chat.messages = chat.messages[:9]
                combined_chat_history.extend(chat.messages)
            
            if not combined_chat_history:
                logging.info("没有对话历史可用于更新persona，跳过更新")
                return
            
            messages = chatbot.build_update_persona_field_prompt(chat_history=combined_chat_history)
            fields = chatbot.get_response(messages=messages, if_add_message=False)
            logging.info(f"Updated fields: {fields}")
            
            current_persona = persona_manager.load_persona(user_id)
            if current_persona:
                updated_persona = update_persona_from_fields(fields, current_persona)
                persona_manager.set_persona(user_id, updated_persona)
                logging.info(f"Updated persona saved for user_id {user_id}")
            else:
                logging.error(f"无法从PersonaManager加载user_id {user_id}的persona数据")
            
            self.accumulated_chat_histories = []
            
        except Exception as e:
            logging.error(f"Error updating persona: {str(e)}")
            print(f"Persona更新失败: {str(e)}")
            self.accumulated_chat_histories = []


def main():
    parser = argparse.ArgumentParser(description="LPA Experiment Runner in Learning Mode")
    
    parser.add_argument("--persona", type=str, required=True, 
                       help="Path to the Persona JSON file")
    parser.add_argument("--persona_db", type=str, required=True,
                       help="Path to the persona database file for persistent updates")
    
    parser.add_argument("--openai_api_key", type=str, required=True,
                       help="OpenAI API key")
    parser.add_argument("--openai_api_base", type=str, required=True,
                       help="OpenAI API base URL")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                       help="Model name (default: gpt-4o-mini)")
    
    parser.add_argument("--update_frequency", type=int, default=3,
                       help="Persona update frequency, updated every k scenes (default: 3)")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of retries (default: 3)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    parser.add_argument("--log_dir", type=str, default="logs_learning",
                       help="Log file directory (default: logs_learning)")
    parser.add_argument("--db_prefix", type=str, default="person_conversations_learning",
                       help="Database file prefix (default: person_conversations_learning)")
    
    args = parser.parse_args()
    
    runner = LearningExperimentRunner(
        persona_file=args.persona,
        persona_db_file=args.persona_db,
        update_frequency=args.update_frequency,
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        model_name=args.model_name,
        log_dir=args.log_dir,
        db_prefix=args.db_prefix,
        max_retries=args.max_retries,
        log_level=args.log_level
    )
    runner.run()


if __name__ == "__main__":
    main() 