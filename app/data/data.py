from pydantic import BaseModel
from datetime import datetime

class BorsaData(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    percentage: float