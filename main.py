"""
PROJECT : LingoBridge
AUTHOR  : PIN CHEN, TSAI
VERSION : v1.2
UPDATE  : 2025-10-28
DETALES :
- 將讀取提問文件並將內容以字串回傳，以便與LBS進行通訊
- 使用方式如下:
```
python main.py --model <modelName> --prompt <filePath>

python main.py --model <modelName> --prompt <filePath> --user <userName>

python main.py --model <modelName> --prompt <fileDir> --user <userName> --outpath <outPath>
```
- 使用範例如下:
```
python main.py --model gemini-2.5-flash --prompt test.txt

python main.py --model gemini-2.5-flash --prompt test.txt --user aino
```
WORKING :
## v1.2
- 追加批量輸出文本
"""

#--- IMPORT--------------------------------------------------------------+

# 模組列表
import argparse     # 標準輸入
import sys          # 系統使用
import os           # 路徑使用

# 自訂功能
from LingoBridge import main as LB_main

#--- STATUS CODE ---------------------------------------------------------+
STATUS_SUCCESS = 0
ERROR_WRITING_FILE = -1

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

def savefile(path, msg, append=False):
    """將文字內容寫入檔案 (預設為覆寫模式)"""
    mode = 'a' if append else 'w'
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, mode, encoding='utf-8') as file:
            file.write(msg.rstrip() + "\n")
        return STATUS_SUCCESS
    except Exception as e:
        print(f"[TFS] ⚠️ 無法寫入檔案 {path}：{e}")
        return ERROR_WRITING_FILE

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
    parser.add_argument("--outpath", default="out.txt", help="輸出文本")
    args = parser.parse_args()

    # lower() 將str中所有英文字母轉換為小寫
    model = args.model.lower()
    path  = os.path.normpath(args.prompt)
    user  = args.user.lower()
    out_path = os.path.normpath(args.outpath)

    # 檢查是否為單一檔案或資料夾
    if os.path.isfile(path):
        print("【LBS】⏱️ 通訊中")
        result = main(model, path, user)
        print("【LBS】✅️ 通訊完成，以下為回覆內容:")
        print(result)

    elif os.path.isdir(path):
        print("【LBS】⏱️ 開始進行批量通訊..")
        # 遍歷整個資料夾
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                print(f"【LBS】📄 處理檔案: {file_path}")
                result = main(model, file_path, user)
                print(result)
                msg = f"{filename} -> {result}"
                savefile(out_path, msg, True)
        print("【LBS】✅️ 已完成所有通訊。")

    else:
        print(f"【LBS】❌ 找不到指定路徑: {path}")
        sys.exit(-1)
