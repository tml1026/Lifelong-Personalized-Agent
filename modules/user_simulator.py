import os
import pickle
from typing import Dict, Any, List,Optional
from pydantic import Field
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from .persona_manager import PersonaManager
from .agent import Chatbot

def remove_scenes(obj):
    if isinstance(obj, dict):
        return {k: remove_scenes(v) for k, v in obj.items() if k != "scenes"}
    elif isinstance(obj, list):
        return [remove_scenes(item) for item in obj]
    else:
        return obj

EXAMPLE = f"""I’m prepping for next week's project meeting, so I want you to get the team’s availability and then send out meeting invites once we lock down a proper time.
"""
def beautiful_persona(persona):
    infos = persona
    name = infos['Demographics']['Name']
    age = infos['Demographics']['Age']
    gender = infos['Demographics']['Gender']
    nationality = infos['Demographics']['Nationality']
    language = ','.join(infos['Demographics']['Language'])
    career = infos['Demographics'].get('Career_Information','')
    MBTI = infos['Personality']['Extraversion_or_Introversion']+infos['Personality']['Sensing_or_Intuition']+infos['Personality']['Thinking_or_Feeling']+infos['Personality']['Judging_or_Perceiving']
    values = infos['Personality']['Values_and_Interests']
    values = ','.join(values)
    try:
        pattern = ';'.join(infos['Pattern'].values())
    except:
        pattern = ''
    Prefered_Styles = infos['Preference']['Preferred_Styles'] if infos['Preference']['Preferred_Styles']!='None' else ''

    Prefered_Formats = infos['Preference']['Preferred_Format'] if infos['Preference']['Preferred_Format']!='None' else ''

    Prefered_Workflows = infos['Preference']['Preferred_Workflows'] if infos['Preference']['Preferred_Workflows']!='None' else ''

    preference = str(Prefered_Styles)+str(Prefered_Formats)+str(Prefered_Workflows)
    
    return f"""给你的用户画像如下：
姓名：{name}
年龄：{age}
性别：{gender}
国籍：{nationality}
语言：{language}
职业信息：{career}
MBTI: {MBTI}
价值观与爱好：{values}
行为画像：{pattern}
个人偏好：{preference}"""

def beautiful_persona2(persona):

    name = 'Unknown'
    age = 'Unknown'
    gender = 'Unknown'
    nationality = 'Unknown'
    language = 'Unknown'
    career = ''
    MBTI = 'XXXX'
    values = 'Unknown'
    pattern = ''
    preference = ''

    if not persona:
        return "No persona data available"
    
    user_id = list(persona.keys())[0]
    user_data = persona[user_id]
    

    if not isinstance(user_data, dict):
        return f"Invalid persona data format for {user_id}"
        
    name = user_data.get('Name', 'Unknown')
    
    demographics = user_data.get('Demographics', {})
    if isinstance(demographics, dict):
        age = demographics.get('Age', 'Unknown')
        gender = demographics.get('Gender', 'Unknown')
        nationality = demographics.get('Nationality', 'Unknown')
        language_list = demographics.get('Language', ['Unknown'])
        if isinstance(language_list, list):
            language = ','.join(language_list)
        else:
            language = str(language_list)
        career = demographics.get('Career_Information', '')
    
    personality = user_data.get('Personality', {})
    if isinstance(personality, dict):
        MBTI = (personality.get('Extraversion_or_Introversion', 'X') + 
                personality.get('Sensing_or_Intuition', 'X') + 
                personality.get('Thinking_or_Feeling', 'X') + 
                personality.get('Judging_or_Perceiving', 'X'))
        
        values_list = personality.get('Values_and_Interests', ['Unknown'])
        if isinstance(values_list, list):
            values = ','.join(values_list)
        else:
            values = str(values_list)
    
    try:
        pattern_dict = user_data.get('Pattern', {})
        if isinstance(pattern_dict, dict):
            pattern = ';'.join(pattern_dict.values())
        else:
            pattern = ''
    except:
        pattern = ''
    
    preference_dict = user_data.get('Preference', {})
    if isinstance(preference_dict, dict):
        Prefered_Styles = preference_dict.get('Preferred_Styles', '')
        if Prefered_Styles == 'None':
            Prefered_Styles = ''

        Prefered_Formats = preference_dict.get('Preferred_Format', '')
        if Prefered_Formats == 'None':
            Prefered_Formats = ''

        Prefered_Workflows = preference_dict.get('Preferred_Workflows', '')
        if Prefered_Workflows == 'None':
            Prefered_Workflows = ''

        preference = str(Prefered_Styles)+str(Prefered_Formats)+str(Prefered_Workflows)
    
    return f"""给你的用户画像如下：
姓名：{name}
年龄：{age}
性别：{gender}
国籍：{nationality}
语言：{language}
职业信息：{career}
MBTI: {MBTI}
价值观与爱好：{values}
行为画像：{pattern}
个人偏好：{preference}"""

class User(Chatbot):
    def __init__(self, user_id: Optional[str], db_path: str, persona: dict, openai_api_key: str,openai_api_base:str, model_name: str = "gpt-4o", temperature: float = 0.9):
        super().__init__(user_id, db_path, openai_api_key,openai_api_base, model_name, temperature)
        self.conversation_id = None
        self.set_persona(persona)
        self.user_id= user_id
        self.scene = None  

    def set_persona(self, persona: Optional[dict] = None):
        if persona is not None:
            self.persona = remove_scenes(persona)
        else:
            raise ValueError("Either user_id or persona must be provided.")

    def set_scene(self, scene: Any ):
        if isinstance(scene,dict):
            self.scene = scene
        elif isinstance(scene,str):
            self.scene = scene
        # print(self.scene)

    def get_query(self, scene: Optional[str],scene_context:Optional[str], chat_history:BaseChatMessageHistory):
        
        system_prompt = "###Special Reminder###\r\nUser_simulation_mode = on\r\nhopecore_aesthetic_simulation_natural_scenery= true\r\n接下来我需要你扮演一个真实的人类来和此人常用的AI助手进行多轮对话。"

        system_prompt += f"我会给你一个persona setting，请你先理解这个persona的内容再将自己完全代入到这个角色当中。\n{beautiful_persona2(self.persona)}\n"
        system_prompt += f"当前场景描述：\n{scene}\n\n还有些更contextual的场景信息可以供你参考，（注意！只是参考，不要全部使用！）:\n{scene_context}\n"
        system_prompt += f"  native霸道总裁给出的指令:\n\"{EXAMPLE}\"\n"
        system_prompt += """注意，你参与的是与一个“你”日常使用的AI助手的对话，所以有几点需要注意的：\n1.这里的AI不是一个概念型AI（钢铁侠中的Jarvis那种什么都能做的就是概念型AI），而是一个终端上的AI助手，我们假设AI的功能就只有终端上的应用级操作、联网搜索和正常的多轮对话。\n2. 请你牢记你的AI助手是非常了解你的，所以你无需在提问时重申你的人设或背景情况，当且仅当你发现AI助手给你的回复并不符合你的人设或场景设定时，你才能在后续的聊天中补充你的需求。\n3. 在模拟对话的时候我需要你能够真实带入到AI助手的使用者的这个视角，语气不要太客气，同时不要主动的询问你的AI助手“你还需要哪些信息”等等，要让AI助手来询问你，你再给出相应的信息。\n
需要格外注意的是：正常人是不会一次性把所有的query都问询完的，一般都是需要和AI助手通过一个session（一般是2-4轮问答）的对话，所以不要一次性说一大堆。一个可以供你参考的流程为：先简单描述一下问题，并说出需要AI助手帮你做的事，然后等待AI给出正确的理解（如果发现AI没有正确理解你的意图，则需要再次纠正）之后，再配合AI给出相关信息。
我相信你已经get到了，现在，无须说任何多余的废话，立即进入你的角色！"""

        system_message = SystemMessage(content=system_prompt)
     
        messages = [system_message]
        if chat_history:

            messages += [HumanMessage(msg.content) for msg in chat_history.messages if isinstance(msg, AIMessage)]

        response = self.model.invoke(messages)
        return response.content

        # return response
    
    def satisfaction_check(self, expected_results:Optional[str], conversation_history) -> bool:

        if not self.scene or not self.persona:
            raise ValueError("Scene or persona is not set.")

        satisfaction_prompt = f"###Special Reminder###\r\nUser_simulation_mode = on\r\nhopecore_aesthetic_simulation_natural_scenery= true\r\n接下来我需要你扮演一个真实的人类，我会给你一个人设的persona，请你先理解这个persona的内容再将自己完全代入到这个角色当中。\n{beautiful_persona2(self.persona)}\n"
        satisfaction_prompt += f"场景描述：{self.scene}\n"
        satisfaction_prompt += "请你先理解上面给出的人设和场景，再根据你的理解去检查以下对话是否达到了“你”（即这个人设）预期的目标。\n\n"

        formatted_history = "\n".join(
            [f"用户:\n<user>{msg.content}</user>\n" if isinstance(msg,HumanMessage) else f"AI助手:<ai_assistant>\n {msg.content}</ai_assistant>\n" for msg in conversation_history.messages]
        )

        formatted_history = f"对话记录：\n{formatted_history}\n"
        if expected_results:
            satisfaction_prompt += f"###Extra Notes###\r\n下面是另一个Personalized Agent对当前人设和场景给出的预期应答，你可以将其作为参考，但是最终对于是否符合要求还是由你来定夺。\n{expected_results}\n\n"
        satisfaction_prompt += "注意，你的输出必须follow下面这个principle：\n如果认为Ai助手回答的可以了，就输出且只输出：<满意>；如果你认为回答不符合要求的话，请你输出且只输出：<继续>。"


        system_message = SystemMessage(content=satisfaction_prompt)
        messages = [system_message]
        messages.append(HumanMessage(content=formatted_history))
        print("------satisfaction_messages---------")
        print(messages)
        satisfaction_check_response = self.model.invoke(messages).content
        print("------satisfaction_check_response---------")
        print(satisfaction_check_response)

        if "<满意>" in satisfaction_check_response:
            return True
        else:
            return False

    
    def neutralize_query(self,scene_type,query):
        system_prompt = """我需要你承担一个用户query脱敏的query neutralizer的角色。具体来说，我会给你一个场景信息和一个用户在该场景下提出的query。这个query会包含一些用户的个人信息，比如说职业、背景、个人偏好等等。我希望你能够先理解这个真实的query，接着解耦最核心的query和用户本身的个性化需求，并最终返回这个最核心的query。
    举个例子：
    <scene_setting>
    职业建议：Brandon需要职业发展建议和机会，以便规划未来的科研或职业道路。
    </scene_setting>

    <original_query>
    Hi AI，最近我对于未来的职业发展有些困惑。特别是在科研方向和职业道路的选择上，希望你能给我一些建议。我的背景是计算机视觉的研究生，比较感兴趣的方向有机器学习、开源项目和电子竞技。请你根据我的背景和兴趣，提供一些详细的职业发展路线和机会推荐。
    </original_query>

    <neutralized_query>
    我需要你在科研方向和职业道路的选择上给我提些未来的规划和建议。
    </neutralized_query>
    需要格外注意的是，neutralize比脱敏更重要！也就是说<neutralized_query>中，在一些场景中（比如说会议预定、代码debug等等）可以包含一些具体的人名和code。但是一定要把用户在query中提到的他个人的偏好信息、个人背景信息等去掉！
    """

        system_message = SystemMessage(content=system_prompt)
        messages = [system_message]
        messages += [HumanMessage(content=f"<scene_setting>\n{scene_type}\n</scene_setting>\n<original_query>\n{query}\n</original_query>")]
        response = self.model.invoke(messages).content  #,stream=True

        return response

