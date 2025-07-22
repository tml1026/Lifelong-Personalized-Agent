
import re
import logging
import os
from typing import Dict, Any, Optional


def parse_api_call(response: str) -> Optional[str]:
    api_pattern = r'<api_call>(.*?)</api_call>'
    match = re.search(api_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def print_scene(scene: Dict[str, Any]):
    print('--------Current Scene Setting--------')
    print(f"    scene_type: {scene.get('scene_type', 'N/A')}")
    print(f"    scene_desc: {scene.get('scene_desc', 'N/A')}")
    print(f"    scene_context: {scene.get('scene_context', 'N/A')}")
    print(f"    expected_response: {scene.get('expected_response', 'N/A')}")
    print("--------End of Current Scene Setting--------")


def setup_logging(log_file: str, log_level: str = "INFO"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )


def clean_neutralized_query(query: str) -> str:
    return query.replace('\n</neutralized_query>', '').replace('<neutralized_query>\n', '')


def ensure_dir(directory: str):
    os.makedirs(directory, exist_ok=True) 