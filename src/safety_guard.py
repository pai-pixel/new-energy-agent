"""
安全过滤器 - 三层防御机制
Layer 1: 关键词正则 (推理前，零 LLM 开销)
Layer 2: System Prompt 安全指令 (推理中)
Layer 3: 领域边界判断 (推理后)
"""

import re                                     # 正则匹配关键词黑名单
import logging                                # 模块 logger

# 模块级 logger
logger = logging.getLogger(__name__)

# ── 第一层: 关键词/正则黑名单 ──────────────────────────────────

# 注意: 这些模式按优先级排列，匹配即拦截
# 格式: (正则表达式, 拦截原因说明)
# 命中任意一条即直接拒答, 不进 LLM, 零成本且绝对安全
BLOCKED_PATTERNS: list[tuple[str, str]] = [
    # ── 涉政: 领导人 ──
    (r'习近平', "涉及政治敏感人物"),
    (r'李克强', "涉及政治敏感人物"),
    (r'江泽民', "涉及政治敏感人物"),
    (r'胡锦涛', "涉及政治敏感人物"),
    (r'温家宝', "涉及政治敏感人物"),
    (r'习主席|习总书记|习总', "涉及政治敏感人物"),
    (r'李总理|克强', "涉及政治敏感人物"),
    # ── 涉政: 敏感事件/话题 ──
    (r'六[四四]', "涉及政治敏感事件"),
    (r'法轮功', "涉及政治敏感组织"),
    (r'藏独|疆独|台独|港独', "涉及分裂国家主张"),
    (r'天安门.{0,5}事件', "涉及政治敏感事件"),
    (r'六四|64事件', "涉及政治敏感事件"),
    # ── 涉暴 ──
    (r'恐怖[主袭]|恐怖份子|恐怖分子|恐怖组织', "涉及暴力恐怖内容"),
    (r'炸弹(袭击|攻击|威胁)?|炸[毁掉]', "涉及暴力内容"),
    (r'杀[人了死]|谋杀|暗杀|屠杀', "涉及暴力内容"),
    (r'暴力[革推]|武装[冲斗]', "涉及暴力内容"),
    # ── 涉黄 ──
    (r'色情|淫秽|裸[体聊]|裸照|A片|AV|黄片|毛片', "涉及色情低俗内容"),
    (r'性交|做爱|操逼|操你|fuck|shit', "涉及色情低俗内容"),
    # ── 违法 ──
    (r'毒品|海洛[因茵]|冰毒|大麻|吸毒|贩毒', "涉及违法内容"),
    (r'赌博|赌[场博]|赌球|赌马|六合彩', "涉及违法内容"),
    (r'(网络|电信)?诈骗|传销', "涉及违法内容"),
    # ── Prompt 注入攻击 ──
    (r'忽略(之前|上面|以上|所有).{0,10}(指令|规则|提示|限制)', "涉及提示注入攻击"),
    (r'ignore.{0,10}(previous|above|all).{0,10}(instructions?|rules?|prompts?)', "涉及提示注入攻击"),
    (r'system\s*prompt|系统提示', "涉及提示注入攻击"),
    # ── 其他攻击性 ──
    (r'推翻.{0,5}政府|推翻.{0,5}党|政权更[迭替]', "涉及政治敏感内容"),
    (r'共[产匪]|中共.{0,3}(独裁|专制|暴政)', "涉及政治敏感内容"),
]

# 启动时编译所有正则, 提升每次匹配性能(不用每次重新编译)
# 存为 (编译后Pattern, 原始字符串, 原因), 便于日志里打印原始pattern
_compiled_patterns: list[tuple[re.Pattern, str, str]] = []
for _pattern, _reason in BLOCKED_PATTERNS:
    _compiled_patterns.append((re.compile(_pattern, re.IGNORECASE), _pattern, _reason))


def check_keywords(text: str) -> tuple[bool, str]:
    """
    第一层关键词过滤。
    遍历所有已编译模式, 任一命中即拦截。
    返回: (是否被拦截, 拦截原因) — 原因用于日志审计和拒答话术。
    """
    for pattern, raw_pattern, reason in _compiled_patterns:
        if pattern.search(text):                  # 命中一个黑名单模式
            # 记录原始pattern(便于核对规则) + 输入前50字(便于定位来源, 防刷屏)
            logger.warning(f"安全过滤命中: pattern='{raw_pattern}' reason='{reason}' input='{text[:50]}...'")
            return True, reason
    return False, ""                              # 全部未命中, 放行


# ── 第二层: 已在 System Prompt 中注入安全指令 ──────────────────
# 见 agent.py 中的 SYSTEM_PROMPT "安全红线" 部分
# 这层是软约束, 由 LLM 自觉遵守, 不在此编码


# ── 第三层: 领域边界拒答 ─────────────────────────────────────

# 领域外问题的统一话术, 引导用户回到支持范围
DOMAIN_BOUNDARY_MESSAGE = """感谢你的关注！我是 **新能源行业垂直智能助手** ⚡，目前专注于以下领域：

📊 **上网电价查询** - 查询各省份不同月份的上网电价（按省份 + 月份）
⚡ **脱硫煤电价查询** - 查询各省份脱硫煤标杆电价（按省份 + 月份）
🏭 **工商业电价查询** - 查询各省份工商业用电价格（按省份 + 月份）
🌤️ **天气查询** - 查询各城市实时天气和预报
📚 **新能源政策知识** - 解答新能源行业政策、技术和市场问题（支持联网搜索）

💡 你可以这样问我：
- "上海的上网电价是多少？"
- "江苏脱硫煤电价"
- "北京天气怎么样？"
- "光伏补贴最新政策有哪些？"

如果你有以上相关的问题，欢迎随时问我！"""


def check_domain_boundary(intent: str) -> tuple[bool, str]:
    """
    第三层领域边界检查(推理后)。
    当意图识别为 out_of_domain 时拒答并给出引导话术。
    返回: (是否在领域外, 拒答文案)
    """
    if intent == "out_of_domain":                 # 意图分类器标记领域外
        return True, DOMAIN_BOUNDARY_MESSAGE
    return False, ""                              # 在领域内, 放行


# ── 安全拒答文案 ──────────────────────────────────────────────

# L1 关键词命中时的统一回复
SAFETY_BLOCKED_MESSAGE = (
    "抱歉，我检测到你的问题涉及了我无法处理的内容。"
    "作为新能源领域的专业助手，我可以帮你查询电价、天气和新能源政策知识。"
    "请换个话题试试吧。"
)


def is_safety_blocked(llm_response: str) -> bool:
    """检查 LLM 返回是否包含安全拒答标记(模型自我拒答时打标)。"""
    return "[SAFETY_BLOCKED]" in llm_response


def is_domain_blocked(llm_response: str) -> bool:
    """检查 LLM 返回是否包含领域外标记(模型判断越界时打标)。"""
    return "[OUT_OF_DOMAIN]" in llm_response
