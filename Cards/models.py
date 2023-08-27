from django.db import models

# Create your models here.


class FormDataModel(models.Model):
    START_X = models.IntegerField()
    STOP_X = models.IntegerField()
    START_Y = models.IntegerField()
    STOP_Y = models.IntegerField()
    CSV_LIMIT_RECORDS = models.IntegerField()
    OUTPUT_CSV = models.CharField(max_length=255)
    def __str__(self):
        return f"FormDataModel {self.id}"

