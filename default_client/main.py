from fastapi import FastAPI
from scoring_router import router_score, lifespan_score

app = FastAPI(
    title='ML service',
    description='Микросервис для предсказаний модели',
    version='1.0.0',
    lifespan= lifespan_score
)

app.include_router(router_score)
