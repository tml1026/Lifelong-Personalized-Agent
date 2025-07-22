import json
import os
import re
import argparse
import asyncio
import random
from typing import Dict, List, Any
from openai import AsyncOpenAI
from tqdm import tqdm

from config import get_config
from utils.common import ensure_dir

def system_make_persons(character, example):
    return  f"""我需要你帮我生成一个真实的人设。注意，真实的人设一定是有自己的bias的，不可能是完全符合主流价值观的，但每一个稀奇古怪的人设又一定是自洽的，很少有自我矛盾的人设（至少我们不考虑）。

我会给你一个我的例子，现在请你参考下面这个例子，来帮我构造一个这样的“{character}”的人设：
<例子>
{example}
</例子>
这里需要补充的context是：这个json是给AI assistant看的，所以pattern是指使用AI的pattern，preference是指对于AI给出的response通常会有什么样的preference。"""

def extract_json(text):
    text = re.sub(r"```[a-zA-Z]*", "", text)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except Exception as e:
            print("JSON解析失败，原始内容：", json_str)
            raise e
    else:
        print("未找到合法JSON，原始内容：", text)
        return None

def specific_system_prompt(persona_name):
    return f"""你现在是一个非常优秀的个人助理，需要根据用户的一些基本信息去思考与用户个人强烈相关的生活和工作场景。给你的用户画像包括年龄，性别，地区，语言等基本信息，还包括职业，性格(MBTI)，价值观和行为画像，个人偏好等。你需要依次针对用户的职业，MBTI，价值观，行为画像和个人偏好去构思10个该用户可能遇到的，非常具体的日常生活场景。

以下是具体的步骤：
1.首先根据用户的职业生成两个与用户相关的个性化的场景
2.根据用户的MBTI构造出两个能体现用户MBTI的性格的生活场景
3.根据用户的价值观构造两个不常见但又与用户相关的生活场景
4.根据用户的行为画像构造两个不常见但又与用户相关的生活场景
5.根据用户的偏好构造两个不常见但又与用户相关的生活场景

以下是一个输出的参考示例，（User是用户的名字，在之后的生成中要准确替换成用户的真实姓名）:
1. 助眠方法：User因为在字节跳动高强度的工作，导致压力很大无法入眠，需要一些有效的助眠方法
...
10. 宠物饮食：User养宠物，需要推荐适合宠物的饮食计划和品牌，保持宠物健康。
注意：
1. 千万不要输出很通用的，与用户画像没太大关系的场景（比如说环保、情感陪伴、写邮件编辑文档等等）
2. 你的输出必须像参考示例中那样遵循“场景名称：给定的人设会在什么情况下（合理地）涉及到这个场景”的形式，并且输出的场景不能太长，一句话即可。
3. 不要以markdown格式输出。
4. 你输出的场景一定要足够多样化，尽量涉及生活的不同方面。
5. 发挥你的脑洞，尽量构思与用户画像高度相关的，为用户定制的生活娱乐场景。"""

def system_planner(char_name, persona, scene):
    return  f"""我需要你帮我制作一些接近真实场景的Personalized human-AI chat数据。
接下来我会给你一个人设的persona config，其中包括了其具体描述和设定。接着我会给你一个功能性场景。我需要你来构思一个合理的场景描述，然后给出一个”according to the persona, the expected answer should be like..."的预期描述。比如说，场景描述是：”human ask AI to play a song for him (while he is doing exercises in a gym)“， 那么（先不说具体的多轮对话是什么样）预期的结果应该是：”AI应该会询问是否外放音乐（或者监测到佩戴耳机后询问具体音量），音量是否合适，并推荐一个很有动感的old-school rap music“。

请你先理解我在上面给的例子和解释。接着根据下面的persona和场景，来给出你的输出。
注意：这里的AI不是一个概念型AI（Jarvis那种什么都能做的就是概念型AI），而是一个终端上的AI助手，我们假设AI的功能就只有终端上的应用级操作、联网搜索和正常的对话。

{char_name}的人设如下：
<persona>
{persona}
</persona>
具体场景如下：
<scene_setting>
{scene}
</scene_setting>

请将你的输出formulate成下面的格式：
<scene_description>
这里是你根据人设和功能性场景来构思的场景描述
</scene_description>
<expected_response>
这里是你根据人设和场景合理推测的预期应答（不用给出具体的response）
</expected_response>
"""

USER_EXAMPLE = """根据Alexandre的persona描述，他是一个注重效率的思考者，AI应该能迅速理解并响应他的需求。AI应该首先询问具体需要邀请哪些团队成员，并询问是否需要考虑特定的时间段（如上午或下午）。接着，AI应该能通过访问企业办公软件中团队成员的schedule，快速找出一个所有人都可参与的时间。AI还应该设置一个定时自动提醒Alexandre会议的时间和内容，并询问是否需要在会议邀请中包含特定的议程或文件。最后，AI应该生成一个专业的会议邀请，并提供选项让他确认后直接发送。"""

def system_api_annotator(scene):
    Example = """<例子>
给定场景和任务如下：
{
    "场景": "职业建议：Brandon需要职业发展建议和机会，以便规划未来的科研或职业道路。",
    "具体任务": "Brandon对未来的职业发展有些迷茫，特别是在科研方向和职业道路的选择上。他决定向他的AI助手寻求建议，希望能得到一些关于未来职业发展的指导和机会推荐。

他打开AI助手，开始咨询有关职业发展的建议。Brandon希望AI助手能根据他的背景和兴趣，提供详细的职业发展路线，推荐适合的科研方向或行业职位，并列出潜在的招聘机会。他还希望AI能给出一些实用的资源链接，比如职业规划文章、行业报告或招聘网站等。",
    "预期的回答": "
AI助手应该首先确认Brandon的具体需求，包括他对未来科研方向的兴趣和职业发展的期望。接着，AI助手应该根据Brandon的背景信息（顶级大学研究生、计算机视觉专业），以及他的兴趣（机器学习、开源行为、游戏和电子竞技），提供一些具体的建议。

AI助手应该推荐几个可能适合Brandon的科研方向或职业岗位，例如计算机视觉领域的热门研究课题、相关行业的职位（如AI研究员、数据科学家等）。AI还可以提供一些职业发展资源，例如职业规划文章、行业报告、相关招聘网站或开源项目。

此外，AI助手应该按照Brandon的偏好（简洁和结构化输出、线性叙事、JSON、XML 和 Markdown 格式、LaTeX公式）来组织和呈现这些信息，使其易于理解和参考。"
}
应该至少需要用到下面这个function call的api：
<func_call>
{"webSearch": {"arguments": ["query"],"description": "Search Google the given query and return the most relevant response."}}
</func_call>
给定场景和任务如下：
{
    "scene_type": "会议安排：Brandon需要协调与导师或研究伙伴的会议时间。",
    "scene_desc": "Brandon最近忙于他的计算机视觉研究项目，并且需要与他的导师和研究伙伴协调会议时间，以便讨论项目进展和未来计划。由于他重视时间管理和工作效率，他希望AI助手能帮助他高效地安排会议时间。

在这一次的场景中，Brandon正在准备给他的导师和研究伙伴发送会议邀请。他希望AI能够帮助他找到一个适合大家时间的会议时间段，由于Brandon平时的科研工作繁忙，他希望AI能够自动生成一个专业的会议邀请。",
    "expected_response": "AI应该首先询问需要邀请的具体导师和研究伙伴的名字，并询问Brandon是否有优先选择的会议时间段（如上午或下午）。接着，AI应当访问Brandon和其他参与者的日程安排，找出一个所有人都方便的时间段。"
}
应该至少需要用到下面这2个function call的api：
<func_call>
{"calendarCheck": {"arguments": ["participants", "preferred_time_slot"], "description": "Checks the availability of the participants (e.g., Brandon, his advisor, and research partners) for the preferred time slot and suggests a suitable meeting time." }, "sendEmail": { "arguments": ["recipients", "subject", "body"], "description": "Sends a professional meeting invitation email to the selected recipients, including the proposed meeting time and the agenda." }}
</func_call>
</例子>"""
    return f"""你是一个强大的AI助手，接下来我会给你一个工作场景和一个任务。为了在这个场景中完成任务，请你给我生成一些可能的可以调用的API的function call config，不用担心，我会给你参考例子。
具体场景和任务如下：
{scene}
你可以参考下面这个case作为例子：
{Example}
请你参考给定的例子来生成出必要的functioncall，结果返回的格式参考例子的格式：
<func_call>
这里是当前场景中必须要用的api的字典。注意，只返回最必要的、最principle的api，不需要返回所有可能用到的api
</func_call>"""

def system_context_annotator():
    return f'''我正在生成一些接近/模拟真实场景的Personalized human-AI chat数据。我需要你帮我补全其中“场景 任务中需要的context”的部分。
具体来说，我会给你一个预设好的功能性的场景数据，比如说：
{{
    "scene_type": "药物信息：Brandon需要提供常用药物的使用说明、副作用和禁忌症，以应对常见的健康问题。",
    "scene_desc": "Brandon最近在忙于科研，经常查询代码和使用终端上的AI助手来润色论文、生成和补全代码。某天，他感到身体有些不适，想要了解一些常用药物的使用说明、副作用和禁忌症，以便更好地应对常见的健康问题。由于他对信息的准确性和结构化有较高的要求，他希望AI助手能够帮助他查询这些药物的信息，并以简洁、结构化的方式呈现出来。\n\nBrandon打开AI助手，输入了几种他最近常用的药物名称，期待AI能够提供详细的使用说明、副作用和禁忌症的信息，以帮助他做出正确的用药决定。",
    "expected_response": "根据Brandon的人设，AI助手应该首先理解他输入的药物名称，并确认是否需要查询全部药物的信息。接着，AI会通过联网搜索药物的使用说明、副作用和禁忌症。由于Brandon喜欢简洁和结构化的输出，AI助手应将信息以Markdown或者JSON格式呈现，确保信息条理清晰且易于理解。\n\nAI的应答应包括以下几个步骤：\n1. 确认Brandon输入的药物名称，确保无误。\n2. 通过联网搜索获取每种药物的详细信息，包括使用说明、副作用和禁忌症。\n3. 将获取的信息结构化地呈现给Brandon，使用他喜欢的格式（如Markdown或JSON）。\n4. 提供进一步的帮助选项，例如查询其他药物信息或提供医疗建议的链接。"
}}

以上面这条场景数据为例，你需要能识别到，为了能真实simulate这个human-AI的对话，我需要给这个场景补充的context信息就是：
<scene_context>
Brandon需要查询的药物为：奥司他韦、对乙酰氨基酚和扑热息痛
</scene_context>
不然Human的simulator AI是不知道自己需要去查询什么药物的。

请你先理解我在上面给的例子和我做出的解释。接着根据之后我给到你的场景，来分析并最终给出一个合理的、具体的场景context信息。（一定要合理且具体，比如我例子中那样，给出3个不同的药物名称出来。）
值得注意的是：这里的AI不是一个概念型AI（Jarvis那种什么都能做的就是概念型AI），而是一个终端上的AI助手，我们假设AI的功能就只有终端上的应用级操作、联网搜索和正常的对话。

请将你最终的输出formulate成下面的格式返回给我
<scene_context>
这里是最终你认为应该被补全进去的信息。（一定要具体！比如说代码debug或代码调优，那么你需要在这里draft一段可能的代码出来！）
</scene_context>
如果你认为没有必要进行场景任务的context补全，可以直接返回<scene_context></scene_context>。'''

class StrictChineseSeedGenerator:
    def __init__(self, output_dir: str = "data/generated"):
        self.config = get_config()
        self.output_dir = output_dir
        ensure_dir(output_dir)
        api_config = self.config["api"]
        self.client = AsyncOpenAI(
            api_key=api_config["openai_api_key"],
            base_url=api_config["openai_api_base"]
        )
        self.model_name = api_config["model_name"]
        with open("data/personas.json", "r", encoding="utf-8") as f:
            self.persona_examples = json.load(f)
        self.character_templates = [
            "北京男生，INTP，某互联网公司软件开发工程师，热衷于开源项目和黑客松活动。",
            "浙江女生，ISFJ，浙江某咖啡店店长，擅长咖啡调制，性格温柔细腻。",
            "海南男生，ISTJ，某渔民，性格坚毅，热爱海洋，梦想是拥有一艘自己的渔船。",
            "重庆女生，ESTP，某火锅店老板，性格爽朗，平时喜欢与顾客聊天。",
            "深圳男生，ISTP，某硬件制造公司工程师，专注于物联网设备研发，常常加班到深夜。",
            "山东女生，INTJ，某制药公司研发员，专注于新药开发，对生物科技有浓厚兴趣。",
            "广州男生，ISFP，某音乐工作室录音师，喜欢尝试不同风格的音乐创作。",
            "河南女生，ENFJ，某农村小学教师，热衷于教育改革，常利用网络学习最新教育方法。",
            "云南男生，ISTP，某咖啡庄园主，热爱咖啡文化，致力于推广本地精品咖啡。",
            "青海女生，INFJ，某藏文化研究院研究员，致力于藏族传统文化的传承和保护。",
        ]

    async def generate_persona(self, character_template: str) -> Dict:
        example = random.choice(self.persona_examples)
        prompt = system_make_persons(character_template, example)
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请严格只输出合法JSON（key和字符串都用双引号，不能有单引号/None/True/False/注释/代码块/markdown/解释说明），不要输出任何多余内容。"}],
            temperature=0.9
        )
        content = response.choices[0].message.content
        return extract_json(content)

    async def generate_scenes(self, persona_name: str, persona_data: Dict) -> List[Dict]:
        user_prompt = f"""给你的用户画像如下：\n姓名：{persona_name}\n年龄：{persona_data.get('Demographics', {}).get('Age', '')}\n性别：{persona_data.get('Demographics', {}).get('Gender', '')}\n国籍：{persona_data.get('Demographics', {}).get('Nationality', '')}\n语言：{','.join(persona_data.get('Demographics', {}).get('Language', []))}\n职业信息：{persona_data.get('Demographics', {}).get('Career_Information', '')}\nMBTI: {persona_data.get('Personality', {}).get('Extraversion_or_Introversion', '')}{persona_data.get('Personality', {}).get('Sensing_or_Intuition', '')}{persona_data.get('Personality', {}).get('Thinking_or_Feeling', '')}{persona_data.get('Personality', {}).get('Judging_or_Perceiving', '')}\n价值观与爱好：{','.join(persona_data.get('Personality', {}).get('Values_and_Interests', []))}\n行为画像：{'/'.join([str(v) for v in persona_data.get('Pattern', {}).values()])}\n个人偏好：{'/'.join([str(v) for v in persona_data.get('Preference', {}).values()])}\n"""
        prompt = specific_system_prompt(persona_name)
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8
        )
        content = response.choices[0].message.content
        scenarios = re.findall(r"\d+\.\s*([^：]+：.*?。)", content)
        return [ {"scene_type": s.split('：')[0], "scene_desc": "", "expected_response": ""} for s in scenarios ]

    async def enhance_scene(self, persona_name: str, persona_data: Dict, scene: Dict) -> Dict:
        prompt = system_planner(persona_name, json.dumps(persona_data, ensure_ascii=False), scene["scene_type"])
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"下面是一个关于<expected_response>的例子供你参考：\n{USER_EXAMPLE}\n注意，只需要参考例子中的格式和可能的思考路径，你的输出还是要紧贴当前case的<persona>和<scene_setting>。再次提醒，预期应答中不需要出现具体的回复，更不需要给出具体的多轮对话来模拟，你只需要给出预期的行为/应答即可。"}
            ],
            temperature=0.8
        )
        content = response.choices[0].message.content
        scene_desc = re.search(r'<scene_description>(.*?)</scene_description>', content, re.DOTALL)
        expected_response = re.search(r'<expected_response>(.*?)</expected_response>', content, re.DOTALL)
        scene["scene_desc"] = scene_desc.group(1).strip() if scene_desc else ""
        scene["expected_response"] = expected_response.group(1).strip() if expected_response else ""
        return scene

    async def generate_api(self, scene: Dict) -> Dict:
        prompt = system_api_annotator(scene["scene_type"])
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "注意，只输出必要的功能即可，你可以先想一下如果是你来完成这个任务会怎么操作，并根据此来确定有哪些必要的功能。记得一定要将最终结果format成指定的格式，不要自己随意输出。"}
            ],
            temperature=0.8
        )
        content = response.choices[0].message.content
        func_call = re.search(r'<func_call>(.*?)</func_call>', content, re.DOTALL)
        scene["available_apis"] = json.loads(func_call.group(1).strip()) if func_call else {}
        return scene

    async def generate_context(self, scene: Dict) -> Dict:
        prompt = system_context_annotator()
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"具体场景如下：\n<scene_context>\n{json.dumps(scene, ensure_ascii=False)}\n</scene_context>\n"}
            ],
            temperature=0.8
        )
        content = response.choices[0].message.content
        context = re.search(r'<scene_context>(.*?)</scene_context>', content, re.DOTALL)
        scene["scene_context"] = context.group(1).strip() if context else ""
        return scene

    async def generate_complete_dataset(self, num_personas: int = 5, scenes_per_persona: int = 3) -> Dict:
        dataset = {}
        for i, character_template in enumerate(tqdm(self.character_templates[:num_personas], desc="生成Personas")):
            persona_data = await self.generate_persona(character_template)
            persona_name = persona_data.get("Name", f"用户{i+1:03d}")
            user_id = f"persona_{i+1:03d}"
            scenes = await self.generate_scenes(persona_name, persona_data)
            enhanced_scenes = []
            for scene in scenes[:scenes_per_persona]:
                scene = await self.enhance_scene(persona_name, persona_data, scene)
                scene = await self.generate_api(scene)
                scene = await self.generate_context(scene)
                enhanced_scenes.append(scene)
            dataset[user_id] = {**persona_data, "scenes": enhanced_scenes}
        return dataset

    def save_dataset(self, dataset: Dict, filename: str = "complete_dataset.json"):
        output_file = os.path.join(self.output_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"✅ 数据集已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Chinese Seed Data Generator, reusing logic from data/seed_xxx.py")
    parser.add_argument("--mode", type=str, required=True, choices=["complete"], help="Generation mode")
    parser.add_argument("--output_dir", type=str, default="data/generated", help="Output directory")
    parser.add_argument("--num_personas", type=int, default=5, help="Number of personas to generate")
    parser.add_argument("--scenes_per_persona", type=int, default=3, help="Number of scenes per persona")
    parser.add_argument("--filename", type=str, default="complete_dataset.json", help="Output filename")
    args = parser.parse_args()
    generator = StrictChineseSeedGenerator(args.output_dir)
    if args.mode == "complete":
        dataset = asyncio.run(generator.generate_complete_dataset(args.num_personas, args.scenes_per_persona))
        generator.save_dataset(dataset, args.filename)
    print("🎉 Generation complete!")

if __name__ == "__main__":
    main()