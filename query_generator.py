import json
import os
from typing import Dict, List, Any
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules import PersonaManager, User
from config import get_config
from utils.common import print_scene, ensure_dir


class QueryGenerator:
    
    def __init__(self, persona_db_path: str, output_dir: str = "data/results_query"):
        self.persona_db_path = persona_db_path
        self.output_dir = output_dir
        self.config = get_config()
        
        ensure_dir(output_dir)
        
        with open(persona_db_path, 'r', encoding='utf-8') as f:
            self.persona_banks = json.load(f)
    
    def process_single_user(self, user_id: str) -> Dict[str, Any]:
        persona_manager = PersonaManager(self.persona_db_path)
        
        persona = persona_manager.load_persona(user_id)
        if not persona:
            raise ValueError(f"Persona for user_id {user_id} not found.")
        
        print(f"Processing user: {user_id}")
        print(persona_manager.print_persona(user_id))
        
        api_config = self.config["api"]
        db_path = self.config["paths"]["chat_history_db"]
        
        user = User(
            user_id=user_id,
            db_path=db_path,
            persona=persona,
            openai_api_key=api_config["openai_api_key"],
            openai_api_base=api_config["openai_api_base"],
            model_name=api_config["model_name"]
        )
        

        scenes = persona.get("scenes", [])
        updated_scenes = []
        
        for scene in scenes:
            if not scene:
                continue
                
            print_scene(scene)
            
            scene_type = scene.get('scene_type', '')
            scene_desc = scene.get("scene_desc", '')
            scene_context = scene.get('scene_context', "")
            
            user.set_scene(scene)
            user_query = user.get_query(scene_desc, scene_context, chat_history=None)
            
            neutralized_query = user.neutralize_query(scene_type, user_query)
            neutralized_query_cleaned = neutralized_query.replace(
                '<neutralized_query>\n', ''
            ).replace('\n</neutralized_query>', '')
            
            scene['personalized_query'] = user_query
            scene['neutralized_query'] = neutralized_query_cleaned
            updated_scenes.append(scene)
        
        updated_persona = persona.copy()
        updated_persona['scenes'] = updated_scenes
        
        return updated_persona
    
    def save_result(self, user_id: str, persona_data: Dict[str, Any], format_type: str = "json"):
        if format_type == "json":
            output_file = os.path.join(self.output_dir, f"{user_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(persona_data, f, ensure_ascii=False, indent=2)
        elif format_type == "jsonl":
            output_file = os.path.join(self.output_dir, f"{user_id}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for scene in persona_data.get('scenes', []):
                    f.write(json.dumps(scene, ensure_ascii=False) + '\n')
    
    def run_batch(self, user_ids: List[str] = None, format_type: str = "json"):
        if user_ids is None:
            user_ids = list(self.persona_banks.keys())
        
        for user_id in user_ids:
            try:
                print(f"\n=== Processing {user_id} ===")
                persona_data = self.process_single_user(user_id)
                self.save_result(user_id, persona_data, format_type)
                print(f"Completed: {user_id}")
            except Exception as e:
                print(f"Error processing {user_id}: {str(e)}")
    
    def run_batch_parallel(self, user_ids: List[str] = None, format_type: str = "json", max_workers: int = 5):
        if user_ids is None:
            user_ids = list(self.persona_banks.keys())
        
        def process_user(user_id):
            try:
                persona_data = self.process_single_user(user_id)
                self.save_result(user_id, persona_data, format_type)
                print(f"Completed: {user_id}")
                return user_id, True, None
            except Exception as e:
                print(f"Error processing {user_id}: {str(e)}")
                return user_id, False, str(e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_user, user_id): user_id for user_id in user_ids}
            
            for future in as_completed(futures):
                user_id, success, error = future.result()
                if not success:
                    print(f"Failed to process {user_id}: {error}")


def main():
    parser = argparse.ArgumentParser(description="Unified Query Generator")
    parser.add_argument("--persona_db", type=str, 
                       default="data/processed_data.json",
                       help="Path to the persona database file")
    parser.add_argument("--output_dir", type=str,
                       default="data/results_query_unified",
                       help="Output directory")
    parser.add_argument("--format", type=str, choices=["json", "jsonl"],
                       default="json", help="Output format")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel mode")
    parser.add_argument("--max_workers", type=int, default=5,
                       help="Maximum number of worker threads (parallel mode only)")
    parser.add_argument("--users", nargs="*", 
                       help="List of user IDs to process. If not specified, all users will be processed.")
    
    args = parser.parse_args()
    
    # Create query generator
    generator = QueryGenerator(
        persona_db_path=args.persona_db,
        output_dir=args.output_dir
    )
    
    if args.parallel:
        generator.run_batch_parallel(
            user_ids=args.users,
            format_type=args.format,
            max_workers=args.max_workers
        )
    else:
        generator.run_batch(
            user_ids=args.users,
            format_type=args.format
        )
    
    print("Query generation complete!")


if __name__ == "__main__":
    main() 