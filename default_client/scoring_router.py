from fastapi import HTTPException, APIRouter
import joblib
import pandas as pd 
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional
import logging
from contextlib import asynccontextmanager
import os

logging.basicConfig(
    level= logging.INFO,
    handlers= [logging.StreamHandler()],
    format= "%(asctime)s - %(levelname)s: %(message)s"
)

logger = logging.getLogger("logger") # создаем объект регистратор 

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'prod_scoring.pkl')

ml_models = {}

@asynccontextmanager 
async def lifespan_score(router):
    try:
        ml_models['Scoring'] = joblib.load(model_path)
        logger.info("Модель успешно загружена")
    except FileNotFoundError:
        logger.critical(f"Не удалось загрузить модель")
        raise
    yield
    ml_models.clear()

router_score = APIRouter(prefix='/scoring')

@router_score.get('/live')
def liveness():
    return {"state": "alive"}


class ClientData(BaseModel):
    Age: int = Field(gt=18)
    Sex: str 
    Job: int = Field(gt=-1, lt=4)
    Housing: str
    Saving_accounts: Optional[str] = Field(default=None, alias="Saving accounts") # потом испраивить логику обработки пропусков
    Checking_account: Optional[str] = Field(default=None, alias="Checking account")
    Credit_amount: int = Field(gt=0, le=50000, alias="Credit amount")
    Duration: int = Field(gt=0)
    Purpose: str

    @field_validator("Saving_accounts", "Checking_account", mode='before')
    @classmethod
    def none_to_str(cls, value):
        if value is None: return "none"
        else: return value

    # проверка на правильность вводимых значений с перехватом исключений
    @field_validator("Sex", "Housing", "Saving_accounts", "Checking_account", "Purpose")
    @classmethod 
    def check_valid_value(cls, value: str, info: ValidationInfo) -> str:
        valid_value = {
            "Sex": {"female", "male"},
            "Housing": {"own", "rent", "free"},
            "Saving_accounts": {"none", "little", "moderate", "quite rich", "rich"},
            "Checking_account": {"none", 'little', 'moderate', 'rich'},
            "Purpose": {'radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'domestic appliances',
            'repairs', 'vacation/others'} 
        }
        value_set = valid_value.get(info.field_name, set())
        if value not in value_set:
            raise ValueError(f"Слова {value} нет в списке допустимых значений, допустимые значения: {value_set}")
        return value
    
threshold = 0.3
    
@router_score.post('/predict')
def predict(data: ClientData):
    try:    
        df = pd.DataFrame([data.model_dump(by_alias=True)])
        proba = ml_models["Scoring"].predict_proba(df)[0]
        default_proba = float(proba[list(ml_models['Scoring'].classes_).index(1)])
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при расчете предсказания")
    else:
        return {
            'default': default_proba,
            'no_default': 1-default_proba,
            "will_default": default_proba>=threshold
        }