"""
SQLAlchemy ORM mirror of database/schema.sql — schema.sql is the source of
truth (run it directly against TimescaleDB for the hypertable setup, which
SQLAlchemy's `create_all()` cannot express). These models exist for
type-safe reads/writes from Python (loaders, the API, training scripts).
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, ForeignKey, Numeric, String, Text,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Exchange(Base):
    __tablename__ = "exchanges"
    exchange_code = Column(String, primary_key=True)
    exchange_name = Column(Text, nullable=False)
    mic = Column(Text, nullable=False)
    country = Column(Text, nullable=False)
    timezone = Column(Text, nullable=False)


class Sector(Base):
    __tablename__ = "sectors"
    sector_code = Column(String, primary_key=True)
    sector_name = Column(Text, nullable=False)
    sector_etf = Column(Text, nullable=False)


class Industry(Base):
    __tablename__ = "industries"
    industry_code = Column(String, primary_key=True)
    industry_name = Column(Text, nullable=False)
    sector_code = Column(String, ForeignKey("sectors.sector_code"), nullable=False)


class Stock(Base):
    __tablename__ = "stocks"
    symbol = Column(String, primary_key=True)
    company_name = Column(Text, nullable=False)
    exchange_code = Column(String, ForeignKey("exchanges.exchange_code"), nullable=False)
    sector_code = Column(String, ForeignKey("sectors.sector_code"), nullable=False)
    industry_code = Column(String, ForeignKey("industries.industry_code"), nullable=False)
    market_cap = Column(Numeric)
    country = Column(Text, nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    date_added = Column(Date)
    date_removed = Column(Date)
    delisted_reason = Column(Text)


class OhlcvBar(Base):
    __tablename__ = "ohlcv_bars"
    symbol = Column(String, primary_key=True)
    session_date = Column(Date, primary_key=True)
    open = Column(Numeric, nullable=False)
    high = Column(Numeric, nullable=False)
    low = Column(Numeric, nullable=False)
    close = Column(Numeric, nullable=False)
    adj_close = Column(Numeric, nullable=False)
    volume = Column(BigInteger, nullable=False)
    source = Column(Text, nullable=False, default="unknown")


class CorporateAction(Base):
    __tablename__ = "corporate_actions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    action_date = Column(Date, nullable=False)
    action_type = Column(Text, nullable=False)
    ratio = Column(Numeric)
    cash_amount = Column(Numeric)
    details = Column(Text)


class MarketFeature(Base):
    __tablename__ = "market_features"
    benchmark = Column(String, primary_key=True)
    session_date = Column(Date, primary_key=True)
    return_1d = Column(Numeric)
    rsi_14 = Column(Numeric)
    volatility_20d = Column(Numeric)
    trend = Column(Text)


class Label(Base):
    __tablename__ = "labels"
    symbol = Column(String, primary_key=True)
    session_date = Column(Date, primary_key=True)
    horizon = Column(String, primary_key=True)
    labeling_method = Column(Text, primary_key=True)
    future_return = Column(Numeric)
    threshold_used = Column(Numeric)
    label = Column(Text, nullable=False)
    target_session_date = Column(Date)
