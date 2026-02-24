from django.db import models


class DefectLog(models.Model):
    timestamp     = models.DateTimeField(auto_now_add=True)
    siniflar      = models.JSONField()   # {'hole': 0.82, 'stain': 0.34, ...}
    aktif_hatalar = models.JSONField()   # ['hole'] — classes that exceeded threshold
    esik          = models.FloatField()
    kaynak        = models.CharField(max_length=20, default='web')

    class Meta:
        ordering = ['-timestamp']
