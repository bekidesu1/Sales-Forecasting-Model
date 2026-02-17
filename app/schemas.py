from pydantic import BaseModel, Field


class SalesFeatures(BaseModel):
    week: int = Field(..., ge=1, le=52, description="Week number in year")
    store_id: int = Field(..., ge=1, le=50, description="Store identifier")
    promo: int = Field(..., ge=0, le=1, description="Promotion flag")
    holiday: int = Field(..., ge=0, le=1, description="Holiday week flag")
    fuel_price: float = Field(..., gt=0)
    cpi: float = Field(..., gt=0)
    unemployment: float = Field(..., ge=0)


class SalesPrediction(BaseModel):
    predicted_weekly_sales: float
