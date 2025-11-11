"""
PROJECT : LingoBridge
AUTHOR  : PIN CHEN, TSAI
VERSION : v2.1
UPDATE  : 2025-10-09
DETALES :
- 與各家LLM進行通訊的程序
- 使用範例如下:
```
$python LingoBridge.py --model gemini-2.5-flash --message "請使用java寫一個Hello World! 測試用程式碼"
```
WORKING :
## v2.1
- 追加使用者功能
- 修改回傳狀態值(status)，以及main()回傳格式。
- 未來可將API key改用環境變數儲存(目前使用items.json 明碼儲存)

STATUS_CODE:
    STATUS_SUCCESS = 0
    ERROR_ITEMS_JSON_MISSING = -101
    ERROR_UNSUPPORTED_MODEL = -201
    ERROR_API_KEY_MISSING = -301
    ERROR_LLM_EXECUTION_FAILED = -302

"""

#--- IMPORT--------------------------------------------------------------+

# 模組列表
import argparse     # 標準輸入
import json         # 讀取JSON檔案
import time         # 時間資訊
from datetime import datetime
import sys          # 系統使用
import os           # 路徑使用

# 各家 API
from openai import OpenAI
import google.generativeai as genai
#from mistralai import Mistral as misai
#import anthropic

#--- VARIABLE------------------------------------------------------------+

# 取得目前檔案所在目錄
# __file__ 表示目前這支程式檔案本身的位置
# os.path.dirname() 取得路徑中的目錄部分
# os.path.normpath() 標準化路徑格式
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 資料路徑
ITEMS_PATH = os.path.join(CURRENT_DIR, 'data', 'items.json')

# 日誌目錄
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
# 日誌路徑
LOG_PATH = os.path.join(LOGS_DIR, 'lists.log')

# 有效使用者清單 (用於建立 log檔案，預設為 default)
VALID_USERS = ["aino", "java"]

#--- INITIAL-------------------------------------------------------------+

# 確保資料夾存在
# data目錄
os.makedirs(os.path.dirname(ITEMS_PATH), exist_ok = True)
# logs目錄
os.makedirs(LOGS_DIR, exist_ok = True)

# 讀取 items.join 資料
if os.path.exists(ITEMS_PATH):
    with open(ITEMS_PATH, 'r', encoding = 'utf-8') as file:
        ITEMS_DATA = json.load(file)
else:
    print("【LBS】⚠️ items.json 不存在，請檢查是否存在於 " + ITEMS_PATH)
    sys.exit(-101)

#--- FUNCTIONS-----------------------------------------------------------+

# 將文字內容儲存為文本檔案 (預設為覆寫模式)
def savefile(path, msg, append=False):
    mode = 'a' if append else 'w'
    try:
        with open(path, mode, encoding = 'utf-8') as file:
            file.write(msg.rstrip() + "\n")
    except Exception as e:
        print(f"【LBS】⚠️ 無法寫入檔案 {path}：{e}")

#--- CLASSES-------------------------------------------------------------+

class BaseLLM:
    """所有 LLM 模型的共同屬性"""
    def __init__(self, model):
        self.model = model      # 模型名稱
        self.reply = ""         # 回覆內容
        self.time  = -1         # 花費時間
        self.max_tokens = 2048  # 設定最大 TOKEN 數
        self.input_tokens  = -1 # 輸入TOKEN數 (int)
        self.output_tokens = -1 # 輸出TOKEN數 (int)
        self.total_tokens  = -1 # 總TOKEN數 (int)

    def log_info(self):
        # 擷取回覆片段
        reply_snippet = (self.reply[:30] + '...') if len(self.reply) > 30 else self.reply
        return f"{self.model} | {self.input_tokens} | {self.output_tokens} | {self.time:.2f}s"

"""OpenAI GPT 系列 (gpt-4o, gpt-5, gpt-5-mini)"""
class ChatGPT(BaseLLM):
    def chat(self, msg):
        # print(f"【LBS】⚠️ 尚未支援該模型: {self.model}")

        # 嘗試初始化設定
        try:
            openai_client = OpenAI(
                api_key=ITEMS_DATA["token_openai"]
            )
        except KeyError:
            print("【LBS】❌ API 密鑰未設定，請檢查Token是否正確。")
            return -301

        # 嘗試進行對話
        try:
            Stime = time.time()         # 開始時間(sec)
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": msg}],
            )
            Etime = time.time()         # 結束時間(sec)
            self.time  = Etime - Stime
            self.reply = response.choices[0].message.content

            # 紀錄 token 數
            # hasattr() 檢查物件是否有指定的屬性，如有就返回 True；否則False。
            if hasattr(response, "usage"):
                self.input_tokens  = response.usage.prompt_tokens
                self.output_tokens = response.usage.completion_tokens
                self.total_tokens  = response.usage.total_tokens
            return 0
        except Exception as e:
            print(f"【LBS】❌ 發生錯誤: {e}")
            return -302

"""Google Gemini 系列 (2.5-pro, 2.5-flash)"""
class Gemini(BaseLLM):
    def chat(self, msg):
        # print(f"【LBS】⚠️ 尚未支援該模型: {self.model}")

        # 嘗試初始化設定
        try:
            # Gemini API Token
            genai.configure(api_key = ITEMS_DATA["token_gemini"])
            model = genai.GenerativeModel(self.model)
        except KeyError:
            print("【LBS】❌ API 密鑰未設定，請檢查Token是否正確。")
            return -301

        # 嘗試進行對話
        try:
            # config
            generation_config = {
                # 生成文本的隨機和多樣性
                # 值越大，文本更具創意和多樣性
                # 值越小，則較接近模型所訓練的文本
                # 數值範圍: 0.0 ~ 2.0
                "temperature": 1,
                # 依據累計機率來限制候選詞彙的範圍
                # 模型會從機率最高的詞彙開始選取
                # 直到累計機率達到設定的值為止
                # 數值範圍: 0.0 ~ 1.0
                "top_p": 0.95,
                # 最大輸出 Token 數限制
                "max_output_tokens": self.max_tokens,
            }

            Stime = time.time()         # 開始時間(sec)
            response = model.generate_content(msg)
            Etime = time.time()         # 結束時間(sec)
            self.time  = Etime - Stime
            self.reply = response.text

            # 紀錄 token 數
            # getattr() 動態獲取物件的屬性值。
            usage = getattr(response, "usage_metadata", None)
            if usage:
                self.input_tokens  = usage.prompt_token_count
                self.output_tokens = usage.candidates_token_count
                self.total_tokens  = usage.total_token_count
            return 0

        except Exception as e:
            print(f"【LBS】❌ 發生錯誤: {e}")
            return -302

"""Mistral 系列"""
class Mistral(BaseLLM):
    def chat(self, msg):
        # print(f"【LBS】⚠️ 尚未支援該模型: {self.model}")

        # 嘗試初始化設定
        try:
            mistral_client = misai(api_key = ITEMS_DATA["token_mistral"])
        except KeyError:
            print("【LBS】❌ API 密鑰未設定，請檢查Token是否正確。")
            return -301

        # 嘗試進行對話
        try:
            Stime = time.time()         # 開始時間(sec)
            response  = mistral_client.chat.complete(
                model = self.model,
                messages = [{"role": "user", "content": msg}],
            )
            Etime = time.time()         # 結束時間(sec)
            self.time  = Etime - Stime
            self.reply = response.choices[0].message["content"]

            # 紀錄 token 數
            # getattr() 動態獲取物件的屬性值。
            usage = getattr(response, "usage", None)
            if usage:
                self.input_tokens  = usage.prompt_tokens
                self.output_tokens = usage.completion_tokens
                self.total_tokens  = usage.total_tokens
            return 0

        except Exception as e:
            print(f"【LBS】❌ 發生錯誤: {e}")
            return -302

"""Anthropic Claude 系列 (Haiku 3.5, Sonnet 3.5...)"""
class Claude(BaseLLM):
    def chat(self, msg):
        # print(f"【LBS】⚠️ 尚未支援該模型: {self.model}")

        # 嘗試初始化設定
        try:
            anthropic_client = anthropic.Anthropic(
                api_key = ITEMS_DATA["token_claude"]
            )
        except KeyError:
            print("【LBS】❌ API 密鑰未設定，請檢查Token是否正確。")
            return -301

        # 嘗試進行對話
        try:
            Stime = time.time()         # 開始時間(sec)
            response = anthropic_client.messages.create(
                model = self.model,
                max_tokens = self.max_tokens,
                messages=[{"role": "user", "content": msg}],
            )
            Etime = time.time()         # 結束時間(sec)
            self.time  = Etime - Stime
            self.reply = response.content[0].text

            # 紀錄 token 數
            # getattr() 動態獲取物件的屬性值。
            usage = getattr(response, "usage", None)
            if usage:
                self.input_tokens  = usage.input_tokens
                self.output_tokens = usage.output_tokens
                self.total_tokens  = usage.input_tokens + usage.output_tokens
            return 0

        except Exception as e:
            print(f"【LBS】❌ 發生錯誤: {e}")
            return -302

#--- MAIN----------------------------------------------------------------+

# 主程序
def main(model, msg, user="default"):

    # 確保為合法使用者
    if user not in VALID_USERS:
        user = "default"

    # 選擇模型
    if model.startswith("gpt"):
        llm = ChatGPT(model)
    elif model.startswith("gemini"):
        llm = Gemini(model)
    elif model.startswith("mistral"):
        llm = Mistral(model)
    elif model.startswith("haiku") or model.startswith("claude"):
        llm = Claude(model)
    else:
        print("【LBS】❌ 不支援的模型名稱")
        return {"status": -201, "error": "unsupported model"}

    # 進行對話
    finished  = llm.chat(msg)
    if finished != 0:
        print("【LBS】❌ LLM非預期輸出，已終止執行")
        return {"status": finished, "error": "llm execution failed"}

    # 建立 log 目錄 (各次提問紀錄)
    # 時間擷取 eg. 250822_184759
    log_time = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir  = os.path.join(LOGS_DIR, user, log_time)
    os.makedirs(log_dir, exist_ok = True)

    # 撰寫日誌
    # 取得當前時間 eg. 2025-08-22 19:25:47
    time_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{time_log}] | " + llm.log_info()
    savefile(LOG_PATH, log_msg, True)
    # 存一份至使用者目錄
    userlog_name = user + ".log"
    userlog_dir  = os.path.join(LOGS_DIR, user)
    userlog_path = os.path.join(userlog_dir, userlog_name)
    savefile(userlog_path, log_msg, True)

    # 儲存提問文本
    prompt_name = "prompt.txt"
    prompt_path = os.path.join(log_dir, prompt_name)
    savefile(prompt_path, msg)

    # 儲存模型回饋
    fb_name = llm.model + ".txt"
    fb_path = os.path.join(log_dir, fb_name)
    savefile(fb_path, llm.reply)
    # print(llm.reply)

    # 結束程序
    return {
        "status": 0,
        "model": llm.model,
        "reply": llm.reply,
    }

#--- ENTRY---------------------------------------------------------------+

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名稱")
    parser.add_argument("--message", required=True, help="要傳送的訊息")
    parser.add_argument("--user", default="default", help="使用者名稱")
    args = parser.parse_args()

    # lower() 將str中所有英文字母轉換為小寫
    model = args.model.lower()
    message = args.message
    user = args.user.lower()

    print("【LBS】⏱️ 通訊中")
    result = main(model, message, user)
    if result["status"] == 0:
        print("【LBS】✅️ 通訊完成")
    sys.exit(result["status"])
