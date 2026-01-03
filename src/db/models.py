from sqlalchemy import Column, Integer, String, Float, DateTime
from .database import Base

class AssetData(Base):
    __tablename__ = "asset_data"

    id = Column(Integer, primary_key=True, index=True)
    asset = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    price = Column(Float)
    log_return = Column(Float)
    return_ = Column(Float)
    volatility_20 = Column(Float)
