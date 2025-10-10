import pyautogui
import time
import sys

print("程序已启动，每10秒将自动点击鼠标左键。")
print("按下 Ctrl+C 即可停止程序。")

try:
    while True:
        # 获取当前鼠标位置
        x, y = pyautogui.position()
        # 在当前位置点击鼠标左键
        x, y = 756, 587
        pyautogui.click(x, y)
        time.sleep(1)
        pyautogui.click(x, y)
        print(f"在 ({x}, {y}) 位置点击了鼠标。")
        # 等待10秒
        time.sleep(10)
except KeyboardInterrupt:
    print("\n程序已停止。")
    sys.exit()
except Exception as e:
    print(f"\n发生错误: {e}")
    print("请确保您已经安装了 pyautogui 库 (pip install pyautogui) 以及相关的依赖。")
    sys.exit()
