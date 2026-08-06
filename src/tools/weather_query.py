"""
天气查询工具 - 使用 wttr.in 免费 API
无需注册，无调用次数限制，支持中文城市名
带 1 小时内存缓存(避免对同一城市反复请求 wttr.in)。
"""

from __future__ import annotations

import logging                                 # 模块 logger
from datetime import datetime, timedelta       # 缓存过期判断(1小时)
from typing import TypedDict, Union            # 类型标注

import httpx                                   # HTTP 客户端调用 wttr.in

# 模块级 logger
logger = logging.getLogger(__name__)

# 简单内存缓存: 城市名 → (天气数据, 缓存时间)
# dict 全局单例, 进程存活期间共享; 多线程下 GIL 保证简单读写安全
_cache: dict[str, tuple[dict, datetime]] = {}


class WeatherResult(TypedDict):
    """天气查询返回结构(类型标注)"""
    city: str                # 城市名
    temp_c: int              # 温度(摄氏度)
    humidity: int            # 湿度(%)
    condition: str           # 天气英文描述
    condition_cn: str        # 天气中文描述
    wind_speed_kmh: int      # 风速(km/h)
    wind_dir: str            # 风向(16方位)
    feels_like_c: int        # 体感温度
    visibility_km: int       # 能见度(km)
    updated_at: str          # 观测时间
    forecast_today: str      # 今日逐时预报简述(可选)


# 天气状况中英文映射
# wttr.in 返回的 weatherDesc 是英文, 这里转成用户友好的中文
CONDITION_MAP = {
    "Sunny": "晴",
    "Clear": "晴",
    "Partly cloudy": "多云",
    "Partly Cloudy": "多云",
    "Cloudy": "阴",
    "Overcast": "阴",
    "Mist": "薄雾",
    "Fog": "雾",
    "Light rain": "小雨",
    "Light Rain": "小雨",
    "Moderate rain": "中雨",
    "Heavy rain": "大雨",
    "Light drizzle": "毛毛雨",
    "Patchy rain possible": "可能有阵雨",
    "Patchy rain nearby": "局部阵雨",
    "Patchy light rain": "局部小雨",
    "Patchy light rain with thunder": "局部雷阵雨",
    "Thunderstorm": "雷阵雨",
    "Moderate or heavy rain with thunder": "中到大雨伴雷",
    "Snow": "雪",
    "Light snow": "小雪",
    "Moderate snow": "中雪",
    "Heavy snow": "大雪",
    "Ice pellets": "冰粒",
    "Light rain shower": "小阵雨",
    "Moderate or heavy rain shower": "中到大阵雨",
    "Torrential rain shower": "暴雨",
    "Light sleet": "小雨夹雪",
    "Light sleet showers": "小冰粒阵雨",
    "Smoke": "烟霾",
    "Haze": "霾",
    "Smoky haze": "烟霾",
    "Sandstorm": "沙尘暴",
    "Dust storm": "沙尘暴",
    "Freezing fog": "冰雾",
    "Blizzard": "暴风雪",
    "Blowing snow": "吹雪",
    "Drizzle": "毛毛雨",
    "Windy": "大风",
}


def query_weather(city: str) -> WeatherResult | dict:
    """
    查询城市天气。
    city: 城市名 (中文或英文，如 "上海", "Shanghai")
    返回 WeatherResult(成功) 或 {"error": ...}(失败)。
    """
    # 1. 检查缓存: 同一城市 1 小时内命中则直接返回, 省一次外呼
    cache_key = city.strip().lower()              # key 归一化(忽略大小写/首尾空格)
    if cache_key in _cache:
        data, cached_at = _cache[cache_key]
        if datetime.now() - cached_at < timedelta(hours=1):  # 未过期
            logger.info(f"天气缓存命中: {city}")
            return data

    # 2. 调用 wttr.in (format=j1 返回 JSON)
    try:
        with httpx.Client(timeout=10) as client:  # 10s 超时, 避免挂死请求
            resp = client.get(
                f"https://wttr.in/{city}",        # 中文城市名 wttr.in 也支持
                params={"format": "j1"},           # j1 = JSON 格式
                headers={"User-Agent": "NewEnergyAgent/1.0"},
            )
            resp.raise_for_status()               # 非200抛异常
            raw = resp.json()
    except Exception as e:
        logger.warning(f"wttr.in 查询 {city} 失败: {e}")
        return {"error": f"天气查询失败: {e}", "city": city}

    # 3. 解析 JSON 结构为 WeatherResult
    try:
        current = raw["current_condition"][0]     # 当前天气(数组第一项)
        weather = raw["weather"][0]               # 今日预报
        hourly = weather.get("hourly", [])        # 逐小时预报

        condition_en = current["weatherDesc"][0]["value"].strip()  # 英文描述
        condition_cn = CONDITION_MAP.get(condition_en, condition_en)  # 中文映射(无则原样)

        # 组装结构化结果
        result: WeatherResult = {
            "city": city,
            "temp_c": int(current["temp_C"]),               # 温度
            "humidity": int(current["humidity"]),           # 湿度
            "condition": condition_en,                      # 英文天气
            "condition_cn": condition_cn,                   # 中文天气
            "wind_speed_kmh": int(current["windspeedKmph"]),  # 风速
            "wind_dir": current["winddir16Point"],          # 风向
            "feels_like_c": int(current["FeelsLikeC"]),     # 体感
            "visibility_km": int(current["visibility"]),    # 能见度
            "updated_at": current["observation_time"],      # 观测时间
        }

        # 附加今日逐时预报简述(可选字段, 增强回复丰富度)
        if hourly:
            result["forecast_today"] = _parse_hourly(hourly)

        # 4. 写入缓存(带时间戳), 下次 1 小时内直接命中
        _cache[cache_key] = (result, datetime.now())
        logger.info(f"天气查询成功: {city} {condition_cn} {result['temp_c']}°C")
        return result

    except (KeyError, IndexError, ValueError) as e:
        # wttr.in 返回结构异常时兜底, 不抛异常给上层
        logger.error(f"天气数据解析失败: {e}")
        return {"error": f"天气数据解析失败，请稍后再试", "city": city}


def _parse_hourly(hourly: list) -> str:
    """
    解析逐小时预报为简短文本。
    取前 8 个时点, 格式: "0时 晴 30°C → 3时 多云 29°C ..."
    """
    if not hourly:
        return ""
    parts = []
    for h in hourly[:8]:                          # 只取前8小时, 控制长度
        raw_time = h.get("time", "0")
        # wttr.in 返回 "0", "100", "200", ..., "2300" 格式 → 转成小时数
        time_val = int(raw_time)
        hour = time_val // 100
        temp = h.get("tempC", "?")                # 温度
        desc_en = h.get("weatherDesc", [{}])[0].get("value", "").strip()  # 英文天气
        desc_cn = CONDITION_MAP.get(desc_en, desc_en)  # 中文映射
        parts.append(f"{hour}时 {desc_cn} {temp}°C")
    return " → ".join(parts)


def format_weather_response(data: WeatherResult | dict) -> str:
    """
    将天气数据格式化为 Markdown 回复文本(供对话层返回给用户)。
    出错时返回带 ⚠️ 的提示。
    """
    if "error" in data:                           # 查询失败分支
        return f"⚠️ {data.get('error')}，城市: {data.get('city', '未知')}"

    w = data
    # 用表格展示核心指标, 易读且美观
    return f"""🌤️ **{w['city']}天气**

| 指标 | 数值 |
|------|------|
| 🌡️ 温度 | **{w['temp_c']}°C**（体感 {w['feels_like_c']}°C）|
| 💧 湿度 | {w['humidity']}% |
| ☁️ 天气 | {w['condition_cn']} |
| 🌬️ 风力 | {w['wind_dir']} {w['wind_speed_kmh']} km/h |
| 👁️ 能见度 | {w['visibility_km']} km |

{f"📅 逐时预报: {w.get('forecast_today', '')}" if w.get('forecast_today') else ""}
🕐 更新时间: {w.get('updated_at', 'N/A')}"""
