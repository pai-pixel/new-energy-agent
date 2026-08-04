"""
天气查询工具 - 使用 wttr.in 免费 API
无需注册，无调用次数限制，支持中文城市名
"""

import logging
from datetime import datetime, timedelta
from typing import TypedDict

import httpx

logger = logging.getLogger(__name__)

# 简单内存缓存 (1小时过期)
_cache: dict[str, tuple[dict, datetime]] = {}


class WeatherResult(TypedDict):
    city: str
    temp_c: int
    humidity: int
    condition: str
    condition_cn: str
    wind_speed_kmh: int
    wind_dir: str
    feels_like_c: int
    visibility_km: int
    updated_at: str


# 天气状况中英文映射
CONDITION_MAP = {
    "Sunny": "晴",
    "Clear": "晴",
    "Partly cloudy": "多云",
    "Partly Cloudy": "多云",
    "Cloudy": "阴",
    "Overcast": "阴",
    "Mist": "雾",
    "Fog": "雾",
    "Light rain": "小雨",
    "Light Rain": "小雨",
    "Moderate rain": "中雨",
    "Heavy rain": "大雨",
    "Light drizzle": "毛毛雨",
    "Patchy rain possible": "可能有阵雨",
    "Patchy light rain": "局部小雨",
    "Thunderstorm": "雷阵雨",
    "Snow": "雪",
    "Light snow": "小雪",
}


def query_weather(city: str) -> WeatherResult | dict:
    """
    查询城市天气
    city: 城市名 (中文或英文，如 "上海", "Shanghai")
    返回 WeatherResult 或错误信息 dict
    """
    # 检查缓存
    cache_key = city.strip().lower()
    if cache_key in _cache:
        data, cached_at = _cache[cache_key]
        if datetime.now() - cached_at < timedelta(hours=1):
            logger.info(f"天气缓存命中: {city}")
            return data

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"https://wttr.in/{city}",
                params={"format": "j1"},
                headers={"User-Agent": "NewEnergyAgent/1.0"},
            )
            resp.raise_for_status()
            raw = resp.json()
    except Exception as e:
        logger.warning(f"wttr.in 查询 {city} 失败: {e}")
        return {"error": f"天气查询失败: {e}", "city": city}

    try:
        current = raw["current_condition"][0]
        weather = raw["weather"][0]
        hourly = weather.get("hourly", [])

        condition_en = current["weatherDesc"][0]["value"]
        condition_cn = CONDITION_MAP.get(condition_en, condition_en)

        result: WeatherResult = {
            "city": city,
            "temp_c": int(current["temp_C"]),
            "humidity": int(current["humidity"]),
            "condition": condition_en,
            "condition_cn": condition_cn,
            "wind_speed_kmh": int(current["windspeedKmph"]),
            "wind_dir": current["winddir16Point"],
            "feels_like_c": int(current["FeelsLikeC"]),
            "visibility_km": int(current["visibility"]),
            "updated_at": current["observation_time"],
        }

        # 加入今日预报简述
        if hourly:
            result["forecast_today"] = _parse_hourly(hourly)

        # 写入缓存
        _cache[cache_key] = (result, datetime.now())
        logger.info(f"天气查询成功: {city} {condition_cn} {result['temp_c']}°C")
        return result

    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"天气数据解析失败: {e}")
        return {"error": f"天气数据解析失败，请稍后再试", "city": city}


def _parse_hourly(hourly: list) -> str:
    """解析逐小时预报为简短文本"""
    if not hourly:
        return ""
    parts = []
    for h in hourly[:8]:  # 只取前8小时
        time_str = h.get("time", "0")[:2]  # 取小时
        temp = h.get("tempC", "?")
        desc_en = h.get("weatherDesc", [{}])[0].get("value", "")
        desc_cn = CONDITION_MAP.get(desc_en, desc_en)
        parts.append(f"{time_str}时 {desc_cn} {temp}°C")
    return " → ".join(parts)


def format_weather_response(data: WeatherResult | dict) -> str:
    """将天气数据格式化为 Markdown 回复文本"""
    if "error" in data:
        return f"⚠️ {data.get('error')}，城市: {data.get('city', '未知')}"

    w = data
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
