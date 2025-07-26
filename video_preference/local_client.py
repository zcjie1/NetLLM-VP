# local_client_example.py

import requests
import json
import time
import random

# --- 1. 配置 ---
SERVER_URL = "http://127.0.0.1:8080"

# 与 run_plm.py 中设置的 SECRET_TOKEN 完全一致
SECRET_TOKEN = "678A0CF4-6357-BBEB-2DE2-AAE887AE1F76"

# --- 2. 模拟的视频信息 ---
SIMULATED_VIDEO_INFO = {
    "name": "BigBuckBunny",
    "height": 540,
    "fps": 25,
    "duration_seconds": 10
}


def get_preference_from_cloud(video_name: str, time_index: int, height: float, fps: float):
    """
    向云服务器发送请求并获取视频偏好决策。

    Args:
        video_name: 视频名
        time_index: 当前帧的时间索引（视频第几秒）
        height: 当前分辨率
        fps: 当前帧率

    Returns:
        一个包含未来三秒偏好值的列表，或者在失败时返回 None。
    """
    # 构造请求头，包含认证 Token 和内容类型
    headers = {
        "Authorization": f"Bearer {SECRET_TOKEN}",
        "Content-Type": "application/json"
    }

    # 构造请求体 (payload)
    payload = {
        "video_name": video_name,
        "time_index": time_index,
        "height": height,
        "fps": fps
    }

    try:
        # 发送 POST 请求，设置10秒超时
        response = requests.post(SERVER_URL, headers=headers, json=payload, timeout=10)

        # 检查响应状态码
        if response.status_code == 200:
            # 请求成功，解析返回的 JSON 数据
            decision = response.json()
            print(f"[{time_index:02d}s] <- 收到决策: {decision}")
            return decision.get("labels")
        else:
            print(f"\033[91m[{time_index:02d}s] <- 请求失败，状态码: {response.status_code}\033[0m")
            try:
                # 尝试打印服务器返回的错误信息
                error_info = response.json()
                print(f"\033[91m  -> 错误详情: {error_info.get('error')}\033[0m")
            except json.JSONDecodeError:
                print(f"\033[91m  -> 无法解析服务器的错误响应。\033[0m")
            return None

    except requests.exceptions.RequestException as e:
        # 处理网络层面的错误 (如连接超时、无法解析主机名等)
        print(f"\033[91m[{time_index:02d}s] <- 网络连接错误: {e}\033[0m")
        return None


def simulate_video_playback():
    """
    主函数，模拟视频播放过程并与服务器交互。
    """
    print("--- 开始模拟本地视频播放客户端 ---")
    print(f"目标服务器: {SERVER_URL}")
    print(f"模拟视频: {SIMULATED_VIDEO_INFO['name']} ({SIMULATED_VIDEO_INFO['height']}p, {SIMULATED_VIDEO_INFO['fps']}fps)")
    print("-" * 35)

    # 模拟从视频第 3 秒开始播放
    start_time = 3
    for current_second in range(start_time, SIMULATED_VIDEO_INFO["duration_seconds"]):
        # 调用函数，与云端交互
        predicted_labels = get_preference_from_cloud(
            video_name=SIMULATED_VIDEO_INFO["name"],
            time_index=current_second,
            height=SIMULATED_VIDEO_INFO["height"],
            fps=SIMULATED_VIDEO_INFO["fps"]
        )

        if predicted_labels:
            print(f"决策应用: 未来3秒的偏好为 [1 = 分辨率, 0 = 帧率] -> {predicted_labels}")
        else:
            print("决策失败，可能采用默认策略")

        # 等待1秒，模拟视频播放到下一秒
        time.sleep(1)
    
    print("\n--- 模拟播放结束 ---")


if __name__ == "__main__":
    simulate_video_playback()