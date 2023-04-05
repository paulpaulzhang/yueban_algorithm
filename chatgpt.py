import openai
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--role", type=str)
parser.add_argument("--content", type=str)
args = parser.parse_args()

# 设置 OpenAI API 密钥
openai.api_key = "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"

example ="""
{"text":"下午三点提醒我去看电影","时间": "下午三点", "待办": "看电影"}
{"text":"明天早上我要去办理健康证","时间": "明天早上", "待办": "办理健康证"}
{"text":"下个月3号交房费","时间": "下个月3号", "待办": "交房费"}
{"text":"情人节要买一束花给女朋友","时间": "情人节", "待办": "买一束花给女朋友"}
{"text":"明天中午去北京市海淀区开会","时间": "明天中午", "待办": "去北京市海淀区开会"}
{"text":"2023年5月12号要去上海参观","时间": "2023年5月12号", "待办": "去上海参观"}
{"text":"二三年六月要交房费","时间": "二三年六月", "待办": "交房费"}
{"text":"明天约朋友去看房","时间": "明天", "待办": "约朋友去看房"}
{"text":"提醒我明天上午要带电脑","时间": "明天上午", "待办": "带电脑"}
{"text":"下周三晚上8点钟开会","时间": "下周三晚上8点钟", "待办": "开会"}
"""

# 使用 GPT-3 模型生成文本
result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",

    messages=[
        {"role": "system", "content": 'You are an office worker who often uses todo software,Please write frequently used to-do or reminder statements and use the json format, such as "text":"下午4点去银行办理业务","time": "下午4点", "todo": "去银行办理业务"}'},
 	      {"role": "user", "content": "给我生成10条"},
        {"role": "assistant", "content": example},
        {"role": "user", "content": "给我生成100条"}
    ],
)

with open('./data.jsonl', 'a') as f:
  f.write(result['choices'][0].message.content)