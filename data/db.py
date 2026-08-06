"""
数据库操作层 - SQLite 电价数据读写
负责电价缓存数据的建表、增删查。
所有 SQL 参数化(? 占位), 防止注入; 每条查询都带异常兜底返回空值。
"""

from __future__ import annotations

import logging                                 # 模块 logger
import sqlite3                                 # SQLite 驱动(标准库)
from pathlib import Path                       # 路径处理

from src.config import DB_PATH                 # 数据库文件路径(由 config 统一管理)

# 模块级 logger
logger = logging.getLogger(__name__)


def get_db() -> sqlite3.Connection:
    """
    获取数据库连接，自动建表(幂等)。
    - 确保父目录存在
    - 打开 WAL 模式(读写并发更稳, 减少文件锁)
    - 建表: 电价记录; 建唯一索引: (省份, 月份, 类型) 保证不重复
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)  # data/ 目录不存在则创建
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")     # WAL 模式: 读写不互相阻塞
    conn.execute("""
        CREATE TABLE IF NOT EXISTS electricity_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,      -- 自增主键
            province VARCHAR(20) NOT NULL,             -- 省份(无后缀, 如 "广东")
            year_month VARCHAR(7) NOT NULL,            -- 月份 "YYYY-MM"
            price_type VARCHAR(30) NOT NULL,           -- 类型英文key(feed_in等)
            price REAL NOT NULL,                       -- 价格(元/千瓦时)
            unit VARCHAR(10) DEFAULT '元/千瓦时',      -- 单位
            source VARCHAR(500),                       -- 数据来源(URL摘要等)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 首次写入时间
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- 最近更新时间
        )
    """)
    # 联合唯一索引: 同一省+月+类型只有一条记录, 重复写入用 REPLACE 覆盖
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_price_unique
        ON electricity_prices(province, year_month, price_type)
    """)
    conn.commit()
    return conn


def query_price(province: str, year_month: str, price_type: str) -> dict | None:
    """
    查询单条电价记录(精确匹配 省+月+类型)。
    返回 {price, unit, source, year_month} 或 None(未命中)。
    """
    try:
        conn = get_db()
        # 参数化查询: WHERE 三个条件完全匹配
        row = conn.execute(
            "SELECT price, unit, source, year_month FROM electricity_prices "
            "WHERE province=? AND year_month=? AND price_type=?",
            (province, year_month, price_type),
        ).fetchone()
        conn.close()
        if row:
            return {"price": row[0], "unit": row[1], "source": row[2], "year_month": row[3]}
    except Exception as e:
        logger.warning(f"电价查询失败: {e}")    # DB异常兜底, 不中断上层
    return None


def query_latest_price(province: str, price_type: str) -> dict | None:
    """
    查询某省某类型的最新电价(按月份倒序取第一条)。
    用途: 精确月份无数据时, 降级返回最近一次记录。
    返回 {price, unit, source, year_month} 或 None。
    """
    try:
        conn = get_db()
        # ORDER BY year_month DESC: 字符串"YYYY-MM"字典序=时间序, 倒序取第一条即最新
        row = conn.execute(
            "SELECT price, unit, source, year_month FROM electricity_prices "
            "WHERE province=? AND price_type=? "
            "ORDER BY year_month DESC LIMIT 1",
            (province, price_type),
        ).fetchone()
        conn.close()
        if row:
            return {"price": row[0], "unit": row[1], "source": row[2], "year_month": row[3]}
    except Exception as e:
        logger.warning(f"最新电价查询失败: {e}")
    return None


def query_trend(province: str, price_type: str, months: int = 12) -> list[dict]:
    """
    查询某省某类电价的历史趋势(按月份升序, 最多 months 条)。
    返回 [{year_month, price, source}, ...]; 无数据或出错返回 []。
    """
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT year_month, price, source FROM electricity_prices "
            "WHERE province=? AND price_type=? "
            "ORDER BY year_month ASC LIMIT ?",   # 升序=从旧到新
            (province, price_type, months),
        ).fetchall()
        conn.close()
        return [{"year_month": r[0], "price": r[1], "source": r[2]} for r in rows]
    except Exception as e:
        logger.warning(f"趋势查询失败: {e}")
        return []


def insert_price(province: str, year_month: str, price_type: str,
                 price: float, unit: str = "元/千瓦时", source: str = "") -> None:
    """
    插入或更新电价记录。
    - INSERT OR REPLACE: 靠唯一索引(省+月+类型)实现"有则覆盖, 无则插入"
    - 幂等: 同一键重复写入安全
    这是"越查越快"缓存方案的核心写入入口(Web搜索结果提取后落库)。
    """
    try:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO electricity_prices "
            "(province, year_month, price_type, price, unit, source, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",  # updated_at 每次更新
            (province, year_month, price_type, price, unit, source),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"电价写入失败: {e}")    # 写库失败不抛, 上层降级继续
        # 注意: 异常时未关闭连接, 依赖 GC 回收(WAL 模式可容忍)


def get_db_stats() -> dict:
    """
    获取数据库统计信息(排查/演示用)。
    返回 {total_records, provinces, by_type}; 异常时返回全零。
    """
    try:
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM electricity_prices").fetchone()[0]  # 总条数
        provinces = conn.execute(                 # 覆盖省份数
            "SELECT COUNT(DISTINCT province) FROM electricity_prices"
        ).fetchone()[0]
        types = conn.execute(                     # 每种类型的记录数
            "SELECT price_type, COUNT(*) FROM electricity_prices GROUP BY price_type"
        ).fetchall()
        conn.close()
        return {
            "total_records": total,
            "provinces": provinces,
            "by_type": {t: c for t, c in types},
        }
    except Exception:
        return {"total_records": 0, "provinces": 0, "by_type": {}}
