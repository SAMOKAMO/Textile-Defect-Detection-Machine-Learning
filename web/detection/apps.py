import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

    def ready(self):
        from . import model_loader
        try:
            model_loader.load()
        except Exception as e:
            logger.critical(
                "Model could not be loaded: %s — server will start "
                "but /api/predict/ will return 503 until the issue is resolved.",
                e,
            )
