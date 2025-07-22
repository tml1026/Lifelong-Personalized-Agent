from typing import Dict, Any, List,Optional
from pydantic import Field
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


example = """"web_search": {
    "arguments": ["query"],
    "description": "Performs a web search using the provided query."
}
"""

class APICaller:
    def __init__(self,openai_api_key: str,openai_api_base:str, model_name: str = "gpt-4o", temperature: float = 0.9):
        self.model = ChatOpenAI(openai_api_key=openai_api_key, openai_api_base=openai_api_base, model=model_name, temperature=temperature)
  
    def api_call_prompt_manager(self,  api_call: str, api_docs:str):

        system_prompt = "我需要你模拟一个在终端上部署的AI助手，你可以访问、操作终端上的应用，以及进行联网搜索等等。当用户给你输入一个api call的时候，我需要你去模拟一下真实的api来返回结果（也就是，假设这个api call被这个API执行了，会拿到什么样的结果）。"
        system_prompt += f"下面是一些常见的api call的description，希望你可以理解这些api的功能，方便你更好地模拟这个api的结果。\n{json.dumps(api_docs,ensure_ascii=False)}"

        messages = [SystemMessage(content=system_prompt)]
        messages.append(HumanMessage(content=api_call))
        return messages

    def get_response(self, messages: List) -> str:
        response = self.model.invoke(messages)
        return response.content

    