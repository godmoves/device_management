from django.db import models


class Device(models.Model):
    device_name = models.CharField(max_length=50)
    device_id = models.CharField(max_length=50)
    device_type = models.CharField(max_length=50)
    device_status = models.CharField(max_length=50)
    device_date = models.DateTimeField('device created')
    device_hist_data = []

    def check_device_status(self):
        return self.device_status

    def __str__(self):
        return self.device_name


class Sensor(models.Model):
    device = models.ForeignKey(Device, on_delete=models.CASCADE)
    sensor_name = models.CharField(max_length=50)
    sensor_id = models.CharField(max_length=50)
    # Current there are 4 types available: hydraulic, bearing, trolley, gear
    sensor_type = models.CharField(max_length=50, default="bearing")
    sensor_value = models.CharField(max_length=50)
    sensor_date = models.DateTimeField('sensor created')

    def __str__(self):
        return self.sensor_name
