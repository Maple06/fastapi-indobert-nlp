# API core module for all endpoints
from fastapi import APIRouter, Form, Query
from typing import List
from .api_v1.endpoints.indobert_endpoint import NLPEndpoint

routerv1 = APIRouter(
    prefix='/api/v1',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@routerv1.post('/')
async def NLPRouteMain(username: str = Form(...), texts: List[str] = Query(...)):
    nlp = NLPEndpoint()
    result = nlp.get_prediction(username, texts)

    return result