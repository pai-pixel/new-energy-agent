"""
新能源智能体 - 配置中心
所有敏感凭证通过环境变量或 Colab Secrets 注入，代码零硬编码 Token。
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Secrets 读取 ──────────────────────────────────────────────


def _get_secret(name: str, required: bool = True) -> str | None:
    """优先级: 环境变量 > Colab Secrets > None"""
    val = os.environ.get(name)
    if val:
        return val
    try:
        from google.colab import userdata  # type: ignore
        val = userdata.get(name)
        if val:
            return val
    except Exception:
        pass
    if required:
        logger.warning(
            f"未找到密钥 '{name}'。请在 Colab 侧边栏 → 🔑 Secrets → 添加 '{name}'，"
            f"或设置环境变量 export {name}=xxx"
        )
    return None


HF_TOKEN = _get_secret("HF_TOKEN", required=False)  # Qwen 模型公开，可留空
NGROK_TOKEN = _get_secret("NGROK_TOKEN", required=False)

# ── 模型配置 ──────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct-AWQ"        # HF Hub 模型 ID
MODEL_CACHE_DIR = "/content/drive/MyDrive/models"  # Colab Drive 模型缓存路径
DATA_DIR = os.environ.get(                          # 数据持久化目录（Drive 优先）
    "NEW_ENERGY_DATA_DIR",
    "/content/drive/MyDrive/new-energy-data"
)
os.makedirs(DATA_DIR, exist_ok=True)                  # 确保目录存在
MODEL_MAX_LEN = 4096                               # vLLM 最大上下文长度
VLLM_PORT = 8000                                   # vLLM API 端口
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"

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

# 城市直接映射回省份的近似
CITY_TO_PROVINCE: dict[str, str] = {v: k for k, v in PROVINCE_TO_CITY.items()}


def get_city_for_province(province: str) -> str:
    """根据省份名获取省会城市名"""
    # 先精确匹配
    city = PROVINCE_TO_CITY.get(province)
    if city:
        return city
    # 模糊匹配：去掉省/市后缀
    for key, val in PROVINCE_TO_CITY.items():
        if province in key or key in province:
            return val
    return province  # 兜底返回原名


def normalize_province(name: str) -> str:
    """标准化省份名称，去掉后缀"""
    for suffix in ["省", "市", "自治区", "壮族自治区", "回族自治区", "维吾尔自治区", "特别行政区"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name
