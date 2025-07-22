from dataclasses import dataclass, Field, asdict
from typing import List,Dict,Tuple, Literal,Optional

# @dataclass
# class str:
#     description: str
#     principles: str
#     examples: Any

@dataclass
class Identity:
    name: str
    user_id: int=Field(default=100000)


@dataclass
class Scene:
    scene_type: str
    sence_desc: str
    expected_results: str
    session: Optional[List[str]]


@dataclass
class Demographics:
    age: int
    gender: Literal['男','女'] 
    nationality: str
    language: List[str]
    career_information: str

@dataclass
class Personality:
    extraversion_introversion: Literal["E","I"]
    sensing_intuition: Literal["S","N"]
    thinking_feeling: Literal["T","F"]
    judging_perceiving: Literal["J","P"]
    values_and_interests: str

@dataclass
class Memory:
    long_term_memory: str
    short_term_memory: str
    faq_feedback_memory: str
    knowledge_base: str


@dataclass
class Pattern:
    behavior_engagement_pattern: str
    usage_pattern: str
    emotion_pattern: str
    purchase_pattern: str

@dataclass
class Preference:
    preferred_styles: str
    preferred_format: str
    preferred_models: str
    preferred_workflows: str

@dataclass
class PersonaConfig:
    identity: Identity
    demographics: Demographics
    personality: Personality
    memory: Memory
    pattern: Pattern
    preference: Preference
    scenes: List[Scene]



def config_to_class(config: Dict) -> PersonaConfig:
    demographics = Demographics(
        age=config['Demographics']['Age'],
        gender=config['Demographics']['Gender'],
        nationality=config['Demographics']['Nationality'],
        language=config['Demographics']['Language'],
        career_information=config['Demographics']['Career_Information']
    )

    personality = Personality(
        extraversion_introversion=config['Personality']['Extraversion_or_Introversion'],
        sensing_intuition=config['Personality']['Sensing_or_Intuition'],
        thinking_feeling=config['Personality']['Thinking_or_Feeling'],
        judging_perceiving=config['Personality']['Judging_or_Perceiving'],
        values_and_interests="; ".join(config['Personality']['Values_and_Interests'])
    )

    # Assuming Memory is optional and not included in the provided example.
    memory = Memory(
        long_term_memory="",
        short_term_memory="",
        faq_feedback_memory="",
        knowledge_base=""
    )

    pattern = Pattern(
        behavior_engagement_pattern=config['Pattern']['Behavior_Engagement_Pattern'],
        usage_pattern=config['Pattern']['Usage_Pattern'],
        emotion_pattern=config['Pattern'].get('Emotion_Pattern', ''),  # Optional handling
        purchase_pattern=config['Pattern']['Purchase_Pattern']
    )

    preference = Preference(
        preferred_styles=config['Preference']['Preferred_Styles'],
        preferred_format=config['Preference']['Preferred_Format'],
        preferred_models=config['Preference'].get('Preferred_Models', '无明显偏好'),  # Optional handling
        preferred_workflows=config['Preference']['Preferred_Workflows']
    )

    scenes = [
        Scene(scene_type=list(scene.keys())[0],expected_response=list(scene.values())[0]['scene_desc'], expected_response=list(scene.values())[0]['expected_response'],session=list(scene.values())[0].get('session',[]))
        for scene in config['scenes']
    ]

    return PersonaConfig(
        demographics=demographics,
        personality=personality,
        memory=memory,
        pattern=pattern,
        preference=preference,
        scenes=scenes
    )


def class_to_config(persona_config: PersonaConfig) -> Dict:
    # Convert dataclass to a nested dictionary
    config_dict = asdict(persona_config)

    # Transform fields to match the expected config format
    config_dict['Demographics'] = {
        'Age': config_dict['demographics']['age'],
        'Gender': config_dict['demographics']['gender'],
        'Nationality': config_dict['demographics']['nationality'],
        'Language': config_dict['demographics']['language'],
        'Career_Information': config_dict['demographics']['career_information']
    }
    
    config_dict['Personality'] = {
        'Extraversion_or_Introversion': config_dict['personality']['extraversion_introversion'],
        'Sensing_or_Intuition': config_dict['personality']['sensing_intuition'],
        'Thinking_or_Feeling': config_dict['personality']['thinking_feeling'],
        'Judging_or_Perceiving': config_dict['personality']['judging_perceiving'],
        'Values_and_Interests': config_dict['personality']['values_and_interests'].split("; ")
    }

    config_dict['Pattern'] = {
        'Behavior_Engagement_Pattern': config_dict['pattern']['behavior_engagement_pattern'],
        'Usage_Pattern': config_dict['pattern']['usage_pattern'],
        'Emotion_Pattern': config_dict['pattern']['emotion_pattern'],
        'Purchase_Pattern': config_dict['pattern']['purchase_pattern']
    }

    config_dict['Preference'] = {
        'Preferred_Styles': config_dict['preference']['preferred_styles'],
        'Preferred_Format': config_dict['preference']['preferred_format'],
        'Preferred_Models': config_dict['preference']['preferred_models'],
        'Preferred_Workflows': config_dict['preference']['preferred_workflows']
    }

    # Convert scenes back into the appropriate nested dictionary format
    config_dict['scenes'] = [
        {scene['scene_type']: {
            'scene_desc': scene['scene_desc'],
            'expected_response': scene['expected_response'],
            'session':scene.get('session',[])
        }} for scene in config_dict['scenes']
    ]

    # Remove the original dataclass keys to avoid redundancy
    for key in ['demographics', 'personality', 'memory', 'pattern', 'preference']:
        del config_dict[key]

    return config_dict