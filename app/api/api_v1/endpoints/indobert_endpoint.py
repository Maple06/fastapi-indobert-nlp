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
            result = nlpService.predict(username, texts)
            return result
        except json.decoder.JSONDecodeError:
            logger.error('Error analysing a text :', "JSONDecodeError at indobert_endpoint.py:15", "Username:", username)
            output = defaultEmptyResult.copy()
            output.update({"error-status": 1, "error-message": r'Error analysing a text: JSONDecodeError at indobert_endpoint.py:15. Make sure the list_content parameter has this format: ["text1", "text2", "text3", ...] (without any backslash \). Double quotes is a necessity.'})
            return output
        except Exception as e:
            logger.error('Error analysing a text :', e, "Username:", username)
            output = defaultEmptyResult.copy()
            output.update({"error-status": 1, "error-message": f"Error analysing a text: {e}"})
            return output