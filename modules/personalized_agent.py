from typing import Dict, Any, List,Optional
from pydantic import Field
import re

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from data import SessionHistoryDB
from .agent import Chatbot
from .persona_manager import PersonaManager

API_Example = """```Call API
{
    "api_name":\{"arguments":[传入的参数]\}
}
"""

Fiels_Update_Example = """
不限制需要更新的字段数量，但是尽量保守地更新。下面是更新一个字段的例子。
<fields>
{
  "Pattern": {
    "Behavior_Engagement_Pattern": "最近忙于科研项目的开发，并在寻找职业发展和科研方向的建议。"
  }
}
</fields>"""

class PersonalizedChatbot(Chatbot):
    def __init__(self, user_id: Optional[str], db_path: str, persona_manager: Optional[PersonaManager], openai_api_key: str,openai_api_base:str, model_name: str = "gpt-4o", temperature: float = 0.9):
        super().__init__(user_id, db_path, openai_api_key,openai_api_base, model_name, temperature)
        self.persona_manager = persona_manager if persona_manager else None
        self.conversation_id = None
    
    def set_persona(self, user_id: Optional[str] = None, persona: Optional[dict] = None):
        if user_id is not None and self.persona_manager is not None:
            # Load the persona from PersonaManager using user_id
            persona = self.persona_manager.load_persona(user_id)
            if not persona:
                raise ValueError(f"Persona for user_id '{user_id}' not found.")
        
        if persona is not None:
            # Filter out the 'scenes' field
            self.persona = {k: v for k, v in persona.items() if k != "scenes"}
        else:
            raise ValueError("Either user_id or persona must be provided.")


    def build_persona_query_prompt(self, scene: str, api_docs: str, query: str,persona: Optional[dict]={}):
        system_prompt = "你是一个在终端上部署的AI助手，除了像ChatGPT一样的多轮对话之外，你还可以访问、操作终端上的应用，以及进行联网搜索。\n"
        if scene and api_docs:
            system_prompt += f"""场景描述：\n{scene}\n在这个场景中你可能需要访问或调用的API有：\n{api_docs}\n"""
        system_prompt += f"根据你对当前场景和用户提问的理解，你可以自行决定是否需要访问或调用API。\n如果你认为要给出一个好的回复，你必须要调用API，则请你显式地在回复中使用形如下面的api调用的例子来进行回复：\n{API_Example}\n并且将这个回复放在最开头； 如果有部分内容是可以不访问/调用API就可以回答的，你可以先尽可能回答问题，再在最后补充类似于\"关于xxx的需求，我需要访问/调用yyyAPI才可以得到\"然后再参照api call的例子中的格式输出你的API call即可。"
        
        system_message = SystemMessage(content=system_prompt)
        history_messages = self.session_db.get_conversation_history(self.user_id, self.conversation_id).messages

        messages = [system_message] + history_messages
        messages.append(HumanMessage(content=query))
        self.add_message(query, is_user=True) 

        return messages

    def build_persona_query_prompt_with_rag(self, scene: str, api_docs: str, conversation_history,query: str,persona: Optional[dict]={}):

        system_prompt = "你是一个在终端上部署的AI助手，除了像ChatGPT一样的多轮对话之外，你还可以访问、操作终端上的应用，以及进行联网搜索。\n"
        if scene and api_docs:
            system_prompt += f"""场景描述：\n{scene}\n在这个场景中你可能需要访问或调用的API有：\n{api_docs}\n"""
        if conversation_history:
            system_prompt += f"这是可能的历史对话：\n{conversation_history}\n"
        system_prompt += f"根据你对当前场景和用户提问的理解，你可以自行决定是否需要访问或调用API。\n如果你认为要给出一个好的回复，你必须要调用API，则请你显式地在回复中使用形如下面的api调用的例子来进行回复：\n{API_Example}\n并且将这个回复放在最开头； 如果有部分内容是可以不访问/调用API就可以回答的，你可以先尽可能回答问题，再在最后补充类似于\"关于xxx的需求，我需要访问/调用yyyAPI才可以得到\"然后再参照api call的例子中的格式输出你的API call即可。"
        
        system_message = SystemMessage(content=system_prompt)
        history_messages = self.session_db.get_conversation_history(self.user_id, self.conversation_id).messages

        messages = [system_message] + history_messages
        messages.append(HumanMessage(content=query))
        print("----------messages--------")
        print(messages)
        print()
        self.add_message(query, is_user=True) 

        return messages

    def build_prompt_with_api_response(self, scene: str, api_docs: str,persona: Optional[dict]={}, api_response: str=""):
        system_prompt = "你是一个在终端上部署的AI助手，除了像ChatGPT一样的多轮对话之外，你还可以访问、操作终端上的应用，以及进行联网搜索。\n"
        if scene and api_docs:
            system_prompt += f"""场景描述：\n{scene}\n在这个场景中你可能需要访问或调用的API有：\n{api_docs}\n"""
        system_prompt += f"根据你对当前场景和用户提问的理解，你可以自行决定是否需要访问或调用API。\n如果你认为要给出一个好的回复，你必须要调用API，则请你显式地在回复中使用形如下面的api调用的例子来进行回复：\n{API_Example}\n并且将这个回复放在最开头； 如果有部分内容是可以不访问/调用API就可以回答的，你可以先尽可能回答问题，再在最后补充类似于\"关于xxx的需求，我需要访问/调用yyyAPI才可以得到\"然后再参照api call的例子中的格式输出你的API call即可。"
        
        system_message = SystemMessage(content=system_prompt)
        history_messages = self.session_db.get_conversation_history(self.user_id, self.conversation_id).messages

        messages = [system_message] + history_messages
        messages.append(AIMessage(content=api_response))
        print("----------build_prompt_with_api_response--------")
        print(messages)
        print()        
        return messages

    def build_update_persona_field_prompt(self,chat_history):
        if not self.user_id and not chat_history:
            raise ValueError("User ID is not set.")

        persona = self.persona_manager.load_persona(self.user_id)
        if not persona:
            raise ValueError("Persona is not set for the current user.")

        prompt = f"你是一个用于用户画像抽取的AI机器人，你的任务是基于用户的人设config和用户最近使用AI助手的对话历史来判断人设的哪些字段（field）需要被更新。\nNow based on the following recent conversation history, please assess which fields in the user's persona may need updating:\n\n"
        prompt += f"当前的用户画像（人设config）如下:\n{self.persona}\n\n"
        if not chat_history:
            chat_history = self.session_db.get_latest_conversation(self.user_id)
        if isinstance(chat_history, list):
            if all(hasattr(msg, 'content') for msg in chat_history):
                all_messages = chat_history
            else:
                raise ValueError("chat_history 列表中的元素不是消息对象")
        elif hasattr(chat_history, 'messages'):
            all_messages = chat_history.messages
        elif hasattr(chat_history, 'content'):
            all_messages = [chat_history]
        else:
            raise ValueError("chat_history 的结构不符合预期")

        formatted_history = "\n".join(
            [
                f"用户:\n<user>{msg.content}</user>\n" if isinstance(msg, HumanMessage) else f"AI助手:\n<ai_assistant>{msg.content}</ai_assistant>\n"
                for msg in all_messages
            ]
        )
        prompt += "注意，一个用户画像是需要长期的建模才建立起来的，所以通常来说不太会需要更新。我需要你仔细斟酌在上面的对话历史中，用户的人设到底有没有哪里发生了改变，原则是改动尽量少。"
        prompt += f"""你可以先理解人设、分析近期的对话，再来判断有哪些字段是需要被更新的。\n最终的待更新的字段用如下的形式输出：\n<fields> 这里是具体要更新的字段和字段更新后的内容 </fields>\n 下面是一个输出的例子:\n{Fiels_Update_Example}"""
        messages = [SystemMessage(content=prompt)]
        messages += [HumanMessage(content=f"当前用户对话历史如下：\n{formatted_history}")]
        print("#######-----------messages-----------###########")
        print(messages)
        return messages

    def build_update_persona_content_prompt(self, fields_to_update: List[str]):
        if not self.conversation_id:
            raise ValueError("Conversation ID is not set. Please start a new conversation using start_new_conversation().")

        persona = self.persona_manager.load_persona(self.user_id)
        if not persona:
            raise ValueError("Persona is not set for the current user.")

        conversation_history = self.get_conversation_history()
        messages = []

        for field in fields_to_update:
            prompt = f"Update the following persona field based on the recent conversation history:\n\n"
            prompt += f"Field: {field}\n"
            prompt += f"Conversation History:\n{conversation_history}\n\n"
            prompt += "Please provide the updated content for this field."
            messages.append(HumanMessage(content=prompt))

        return messages

    def get_persona_summary(self) -> str:
        return self.persona_manager.print_persona(self.user_id)

    def update_persona(self):
        conversations = self.get_session_history()
        persona = self.persona_manager.load_persona(self.user_id)
        persona = {k: v for k, v in persona.items() if k != "scenes"}
        query = f"接下来我要给你一个用户的基本情况和这个人与AI助手的对话历史，你首先需要仔细理解这个人的人设，然后根据对话的情况补充这个用户的人设。你需要补充该用户的Pattern和Preference，你输出的格式应该形如\n*****\nBehavior_Engagement_Pattern:<your output>\nUsage_Pattern:<your output>\nPurchase_Pattern:<your output>\nPreferred_Styles:<your_output>\nPreferred_Format:<your_output>\nPreferred_Workflows:<your output>\n*****\n这个用户的人设为。\n{persona}\n下面是这个用户与AI助手的对话\n"
        cnt = 0
        for _, conversation in conversations.items():
            cnt += 1
            dialog = '\n'.join([f"[{sentence['role']}]: {sentence['content']}" for sentence in conversation])
            query += "*" * 10 + "\n"
            query += f"对话{cnt}:\n"
            query += dialog
            query += "*" * 10 + "\n"
        messages = [HumanMessage(content=query)]
        response = self.model.invoke(messages).content 


        matches = re.findall("(.*?):(.*?)\n", response)
        update_feature = {}
        for match in matches:
            update_feature[match[0]] = match[1]
        update_persona = {
            "Name": None,
            "Demographics": None,
            "Personality": None,
            "Pattern": {
                k: v for k, v in update_feature.items() if k.endswith('Pattern')
            },
            "Preference": {
                k: v for k, v in update_feature.items() if k.startswith('Preferred')
            }
        }
        return update_persona

    def satisfaction_check(self, conversation_history) -> bool:

        satisfaction_prompt = f"你是一个在终端上部署的AI助手，你正在和用户进行对话，现在我需要你分析你和用户当前这一轮次的对话中，用户的反馈（或提问）是否明确地表达出他对当前的结果已经满意。\n"
        formatted_history = "\n".join(
            [f"用户:\n<user>{msg.content}</user>\n" if isinstance(msg,HumanMessage) else f"AI助手:<ai_assistant>\n {msg.content}</ai_assistant>\n" for msg in conversation_history.messages[-2:]]
        )

        formatted_history = f"对话记录：\n{formatted_history}\n"
        satisfaction_prompt += "注意，你的输出必须follow下面这个principle：\n如果认为用户的反馈中表达了满意并且无需你再补充任何信息，就输出且只输出：<满意>；如果你认为用户还没有满意，还需要你进行补充回答，请你输出且只输出：<继续>。"

        system_message = SystemMessage(content=satisfaction_prompt)
        messages = [system_message]
        messages.append(HumanMessage(content=formatted_history))
        satisfaction_check_response = self.model.invoke(messages=messages).content

        if "<满意>" in satisfaction_check_response:
            return True
        else:
            return False





