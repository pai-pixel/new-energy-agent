"""
新能源智能体 - 配置中心
所有敏感凭证通过环境变量注入，代码零硬编码 Token。
模型层: DeepSeek API (OpenAI 兼容)
"""

import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Secrets 读取 ──────────────────────────────────────────────


def _get_secret(name: str, required: bool = True) -> Optional[str]:
    """优先级: 环境变量 > .env 文件"""
    val = os.environ.get(name)
    if val:
        return val
    # 尝试从 .env 文件加载
    try:
        from pathlib import Path
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == name:
                            val = v.strip().strip('"').strip("'")
                            if val:
                                os.environ[name] = val
                                return val
    except Exception:
        pass
    if required:
        logger.warning(
            f"未找到密钥 '{name}'。请设置环境变量 export {name}=xxx "
            f"或在项目根目录 .env 文件中添加 {name}=xxx"
        )
    return None


# ── DeepSeek API 配置 ─────────────────────────────────────────

DEEPSEEK_API_KEY = _get_secret("DEEPSEEK_API_KEY", required=False)
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 兼容旧的环境变量名
LLM_API_KEY = DEEPSEEK_API_KEY or _get_secret("LLM_API_KEY", required=False) or "your-deepseek-api-key-here"
LLM_BASE_URL = DEEPSEEK_BASE_URL
LLM_MODEL = DEEPSEEK_MODEL

# ngrok (Colab 部署用)
NGROK_TOKEN = _get_secret("NGROK_TOKEN", required=False)

# ── 模型参数 ──────────────────────────────────────────────────

MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "4096"))
MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.7"))

# ── 数据路径 ──────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NEW_ENERGY_DATA_DIR", os.path.join(PROJECT_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "electricity_prices.db")

# ── 电价类型映射 ──────────────────────────────────────────────

PRICE_TYPE_MAP = {
    "feed_in": "上网电价",
    "desulfurized_coal": "脱硫煤电价",
    "commercial_industrial": "工商业电价",
}

PRICE_TYPE_REVERSE = {v: k for k, v in PRICE_TYPE_MAP.items()}

# ── 省份 → 省会/主要城市映射 (天气查询用) ─────────────────────

PROVINCE_TO_CITY: dict[str, str] = {
    "北京": "北京", "北京市": "北京",
    "上海": "上海", "上海市": "上海",
    "天津": "天津", "天津市": "天津",
    "重庆": "重庆", "重庆市": "重庆",
    "广东": "广州", "广东省": "广州",
    "江苏": "南京", "江苏省": "南京",
    "浙江": "杭州", "浙江省": "杭州",
    "山东": "济南", "山东省": "济南",
    "河南": "郑州", "河南省": "郑州",
    "四川": "成都", "四川省": "成都",
    "湖北": "武汉", "湖北省": "武汉",
    "湖南": "长沙", "湖南省": "长沙",
    "福建": "福州", "福建省": "福州",
    "安徽": "合肥", "安徽省": "合肥",
    "河北": "石家庄", "河北省": "石家庄",
    "辽宁": "沈阳", "辽宁省": "沈阳",
    "陕西": "西安", "陕西省": "西安",
    "江西": "南昌", "江西省": "南昌",
    "广西": "南宁", "广西壮族自治区": "南宁",
    "山西": "太原", "山西省": "太原",
    "云南": "昆明", "云南省": "昆明",
    "贵州": "贵阳", "贵州省": "贵阳",
    "吉林": "长春", "吉林省": "长春",
    "黑龙江": "哈尔滨", "黑龙江省": "哈尔滨",
    "甘肃": "兰州", "甘肃省": "兰州",
    "海南": "海口", "海南省": "海口",
    "内蒙古": "呼和浩特", "内蒙古自治区": "呼和浩特",
    "宁夏": "银川", "宁夏回族自治区": "银川",
    "青海": "西宁", "青海省": "西宁",
    "西藏": "拉萨", "西藏自治区": "拉萨",
    "新疆": "乌鲁木齐", "新疆维吾尔自治区": "乌鲁木齐",
    "台湾": "台北", "台湾省": "台北",
    "香港": "香港", "香港特别行政区": "香港",
    "澳门": "澳门", "澳门特别行政区": "澳门",
}

CITY_TO_PROVINCE: dict[str, str] = {v: k for k, v in PROVINCE_TO_CITY.items()}

# 31 个省级行政区（不含港澳台用于电价查询）
MAINLAND_PROVINCES: list[str] = [
    "北京", "上海", "天津", "重庆",
    "广东", "江苏", "浙江", "山东", "河南", "四川",
    "湖北", "湖南", "福建", "安徽", "河北", "辽宁",
    "陕西", "江西", "广西", "山西", "云南", "贵州",
    "吉林", "黑龙江", "甘肃", "海南", "内蒙古", "宁夏",
    "青海", "西藏", "新疆",
]


def get_city_for_province(province: str) -> str:
    """根据省份名获取省会城市名"""
    city = PROVINCE_TO_CITY.get(province)
    if city:
        return city
    for key, val in PROVINCE_TO_CITY.items():
        if province in key or key in province:
            return val
    return province


def normalize_province(name: str) -> str:
    """标准化省份名称，去掉后缀"""
    for suffix in ["省", "市", "自治区", "壮族自治区", "回族自治区", "维吾尔自治区", "特别行政区"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name
