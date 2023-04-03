# API core module for all endpoints
from fastapi import APIRouter, Form
from .api_v1.endpoints.indobert_endpoint import NLPEndpoint

routerv1 = APIRouter(
    prefix='/api/v1',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@routerv1.post('/')
async def NLPRouteMain(username: str = Form(...), list_content: str = Form(...)):
    nlp = NLPEndpoint()
    result = nlp.get_prediction(username, list_content)

    return result