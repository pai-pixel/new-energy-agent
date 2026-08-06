"""
新能源智能体 - 配置中心
所有敏感凭证通过环境变量注入，代码零硬编码 Token。
模型层: DeepSeek API (OpenAI 兼容)

日志说明: 本模块不再自己配置 logging.basicConfig,
统一由 src/logging_config 管理(控制台+文件双输出+请求ID)。
"""

# ── 标准库导入 ─────────────────────────────────────────────────
import os                                    # 读取环境变量 / 拼接路径
from pathlib import Path                      # 跨平台路径处理
from typing import Optional                   # 类型标注: 可能为 None 的返回值

# ── 日志: 使用统一日志配置 ─────────────────────────────────────
# 先导入 logging_config 触发其初始化(幂等), 再拿统一 logger。
# 注意: 必须先于其他模块 import, 否则后续模块的日志没有 handler 输出。
from src.logging_config import init_logging, logger
init_logging()                                # 确保根 logger 已配置(幂等)

# ── Secrets 读取 ──────────────────────────────────────────────


def _get_secret(name: str, required: bool = True) -> Optional[str]:
    """
    读取密钥配置, 优先级: 环境变量 > .env 文件。
    - 环境变量优先(部署时常用 export 注入)
    - 没有环境变量则解析项目根目录的 .env 文件(本地开发常用)
    - required=True 且两处都没有时, 打 warning 提示
    """
    val = os.environ.get(name)                # 第一步: 读环境变量
    if val:                                   # 环境变量有值就直接用
        return val

    # 第二步: 尝试从 .env 文件加载
    try:
        env_file = Path(__file__).resolve().parent.parent / ".env"  # 项目根目录 .env
        if env_file.exists():                 # 文件存在才读
            for line in env_file.read_text(encoding="utf-8").splitlines():
                # 跳过空行、注释行、没有 = 的行
                if not line.strip() or line.strip().startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)     # 按第一个 = 拆成 key/value
                if k.strip() == name:         # key 匹配到目标
                    val = v.strip().strip('"').strip("'")  # 去掉值两侧空白和引号
                    if val:                   # 有值才返回
                        os.environ[name] = val  # 顺手写回环境变量, 后续直接可用
                        return val
    except Exception:
        pass                                  # 解析 .env 失败静默降级(不阻断启动)

    # 第三步: 都找不到
    if required:
        logger.warning(
            f"未找到密钥 '{name}'。请设置环境变量 export {name}=xxx "
            f"或在项目根目录 .env 文件中添加 {name}=xxx"
        )
    return None                               # 非必需密钥返回 None


# ── DeepSeek API 配置 ─────────────────────────────────────────
# required=False: 启动时不强制要求, 真正调用 API 时 model_engine 会二次校验
DEEPSEEK_API_KEY = _get_secret("DEEPSEEK_API_KEY", required=False)  # API 密钥
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # 官方端点
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")  # 默认模型

# 兼容旧的环境变量名(早期版本叫 LLM_*), 保证老配置还能用
LLM_API_KEY = DEEPSEEK_API_KEY or _get_secret("LLM_API_KEY", required=False) or "your-deepseek-api-key-here"
LLM_BASE_URL = DEEPSEEK_BASE_URL
LLM_MODEL = DEEPSEEK_MODEL

# ngrok 隧道令牌(Colab 在线演示用, 本地无需配置)
NGROK_TOKEN = _get_secret("NGROK_TOKEN", required=False)

# ── 模型参数 ──────────────────────────────────────────────────
MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "4096"))  # 单次生成上限
MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.7"))  # 采样温度

# ── 数据路径 ──────────────────────────────────────────────────
# PROJECT_DIR: 项目根目录(本文件在 src/ 下, 往上一级即根目录)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR 可被环境变量覆盖(部署时数据放独立磁盘的场景)
DATA_DIR = os.environ.get("NEW_ENERGY_DATA_DIR", os.path.join(PROJECT_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)          # 目录不存在则创建
DB_PATH = os.path.join(DATA_DIR, "electricity_prices.db")  # SQLite 数据库文件路径

# ── 电价类型映射 ──────────────────────────────────────────────
# 英文 key(系统内部/Database 存储) → 中文名(用户可见/搜索关键词)
PRICE_TYPE_MAP = {
    "feed_in": "上网电价",                      # 燃煤/新能源标杆上网电价
    "desulfurized_coal": "脱硫煤电价",          # 燃煤机组脱硫标杆电价
    "commercial_industrial": "工商业电价",      # 工商业用电目录电价
}
# 反向映射: 中文名 → 英文 key, 便于按中文反查
PRICE_TYPE_REVERSE = {v: k for k, v in PRICE_TYPE_MAP.items()}

# ── 省份 → 省会/主要城市映射 (天气查询用) ─────────────────────
# 用户说"山东天气"时, 定位到省会"济南"去查 wttr.in
# 同时收录带后缀的写法(山东省/北京市), 免去先 normalize 的麻烦
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

# 反向: 城市名 → 省份名(某些场景用户直接报城市)
CITY_TO_PROVINCE: dict[str, str] = {v: k for k, v in PROVINCE_TO_CITY.items()}

# 31 个省级行政区(不含港澳台用于电价查询 — 电价数据源不覆盖港澳台)
MAINLAND_PROVINCES: list[str] = [
    "北京", "上海", "天津", "重庆",
    "广东", "江苏", "浙江", "山东", "河南", "四川",
    "湖北", "湖南", "福建", "安徽", "河北", "辽宁",
    "陕西", "江西", "广西", "山西", "云南", "贵州",
    "吉林", "黑龙江", "甘肃", "海南", "内蒙古", "宁夏",
    "青海", "西藏", "新疆",
]


def get_city_for_province(province: str) -> str:
    """
    根据省份名获取省会城市名(天气查询用)。
    先精确查表, 查不到再做包含匹配(容忍"广东啊"这种带口语的输入), 最后原样返回。
    """
    city = PROVINCE_TO_CITY.get(province)     # 精确匹配
    if city:
        return city
    for key, val in PROVINCE_TO_CITY.items():  # 模糊包含匹配
        if province in key or key in province:
            return val
    return province                           # 兜底: 把省份名当城市名用


def normalize_province(name: str) -> str:
    """
    标准化省份名称, 去掉"省/市/自治区"等后缀。
    例: "广东省" → "广东", "内蒙古自治区" → "内蒙古"
    数据库里存的是无后缀短名, 查询前必须归一化。
    """
    # 依次尝试去掉各类行政区后缀; 注意长后缀要先匹配(维吾尔自治区 在 自治区 之前)
    for suffix in ["省", "市", "自治区", "壮族自治区", "回族自治区", "维吾尔自治区", "特别行政区"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]       # 切除后缀
            break                             # 只切一个后缀即可
    return name
