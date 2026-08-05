"""
数据库操作层 - SQLite 电价数据读写
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src.config import DB_PATH, logger


def get_db() -> sqlite3.Connection:
    """获取数据库连接，自动建表"""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS electricity_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            province VARCHAR(20) NOT NULL,
            year_month VARCHAR(7) NOT NULL,
            price_type VARCHAR(30) NOT NULL,
            price REAL NOT NULL,
            unit VARCHAR(10) DEFAULT '元/千瓦时',
            source VARCHAR(500),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_price_unique
        ON electricity_prices(province, year_month, price_type)
    """)
    conn.commit()
    return conn


def query_price(province: str, year_month: str, price_type: str) -> dict | None:
    """查询单条电价记录"""
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT price, unit, source, year_month FROM electricity_prices "
            "WHERE province=? AND year_month=? AND price_type=?",
            (province, year_month, price_type),
        ).fetchone()
        conn.close()
        if row:
            return {"price": row[0], "unit": row[1], "source": row[2], "year_month": row[3]}
    except Exception as e:
        logger.warning(f"电价查询失败: {e}")
    return None


def query_latest_price(province: str, price_type: str) -> dict | None:
    """查询某省某类型最新电价"""
    try:
        conn = get_db()
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
    """查询某省某类电价的历史趋势（按月份升序）"""
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT year_month, price, source FROM electricity_prices "
            "WHERE province=? AND price_type=? "
            "ORDER BY year_month ASC LIMIT ?",
            (province, price_type, months),
        ).fetchall()
        conn.close()
        return [{"year_month": r[0], "price": r[1], "source": r[2]} for r in rows]
    except Exception as e:
        logger.warning(f"趋势查询失败: {e}")
        return []


def insert_price(province: str, year_month: str, price_type: str,
                 price: float, unit: str = "元/千瓦时", source: str = "") -> None:
    """插入或更新电价记录"""
    try:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO electricity_prices "
            "(province, year_month, price_type, price, unit, source, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (province, year_month, price_type, price, unit, source),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"电价写入失败: {e}")


def get_db_stats() -> dict:
    """获取数据库统计信息"""
    try:
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM electricity_prices").fetchone()[0]
        provinces = conn.execute(
            "SELECT COUNT(DISTINCT province) FROM electricity_prices"
        ).fetchone()[0]
        types = conn.execute(
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
