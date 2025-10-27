"""
PROJECT : LingoBridge
AUTHOR  : PIN CHEN, TSAI
VERSION : v1.0
UPDATE  : 2025-10-09
DETALES :
- 將讀取提問文件並將內容以字串回傳，以便與LBS進行通訊
- 使用範例如下:
```
$python main.py --model gemini-2.5-flash --prompt test.txt
```
```
$python main.py --model gemini-2.5-flash --prompt test.txt --user aino
```
WORKING :
"""

#--- IMPORT--------------------------------------------------------------+

# 模組列表
import argparse     # 標準輸入
import sys          # 系統使用
import os           # 路徑使用

# 自訂功能
from LingoBridge import main as LB_main

#--- VARIABLE------------------------------------------------------------+

# 目前檔案所在目錄
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

#--- INITIAL-------------------------------------------------------------+

#--- FUNCTIONS-----------------------------------------------------------+

# 文本檔案內容以字串回傳
def readfile(path):
    try:
        # errors = 'replace' 把無法解碼的字元換成 �(U+FFFD)。
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read()
    except Exception as e:
        print(f"【fR】⚠️ 無法讀取檔案 {path}：{e}")

#--- MAIN----------------------------------------------------------------+

# 主程序
def main(model, path, user="default"):

    # 讀取文本
    message = readfile(path)

    # 呼叫 LBS
    response = LB_main(model, message, user)

    # 回傳結果文本
    if response["status"] == 0:
        return response["reply"]
    else:
        print("【LBS】❌ 錯誤代碼: " + str(response["status"]))
        return ""

#--- ENTRY---------------------------------------------------------------+

if __name__ == "__main__":
    # 標準輸入
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名稱")
    parser.add_argument("--prompt", required=True, help="檔案路徑")
    parser.add_argument("--user", default="default", help="使用者名稱")
    args = parser.parse_args()

    # lower() 將str中所有英文字母轉換為小寫
    model = args.model.lower()
    path  = os.path.normpath(args.prompt)
    user  = args.user.lower()

    print("【LBS】⏱️ 通訊中")
    result = main(model, path, user)
    print("【LBS】✅️ 通訊完成，以下為回覆內容:")
    print(result)
