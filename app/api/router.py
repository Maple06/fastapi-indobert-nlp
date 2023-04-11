# API core module for all endpoints
from fastapi import APIRouter, Form, Request
from .api_v1.endpoints.indobert_endpoint import NLPEndpoint

routerv1 = APIRouter(
    prefix='/api/v1',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@routerv1.post('/')
async def NLPRouteMain(data : Request):
    nlp = NLPEndpoint()
    req_data = await data.json()
    result = nlp.get_prediction(req_data["username"], req_data["list_content"])

    return result