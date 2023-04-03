import json

from ....core.logging import logger
from ..services.indobert_service import NLPService
from ...load_models import defaultEmptyResult

# Module of an endpoint
class NLPEndpoint:
    def __init__(self):
        pass

    def get_prediction(self, username, texts):
        try:
            nlpService = NLPService()
            texts = json.loads(texts)
            result = nlpService.predict(username, texts)
            return result

        except Exception as e:
            logger.error('Error analysing a text :', e, "Username:", username)
            return defaultEmptyResult.update({"error-status": 1, "error-message": f"Error analysing an image: {e}"})