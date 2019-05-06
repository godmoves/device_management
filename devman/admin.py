from django.contrib import admin

from .models import Device, Sensor


class SensorInline(admin.TabularInline):
    model = Sensor
    extra = 0


class DeviceAdmin(admin.ModelAdmin):
    fields = ['device_name',
              'device_id',
              'device_type',
              'device_status',
              'device_date']
    inlines = [SensorInline]

    list_display = ('device_name', 'device_id', 'device_type', 'device_status', 'device_date')

    list_filter = ['device_status']

    search_fields = ['device_name']


class SensorAdmin(admin.ModelAdmin):
    list_display = ('sensor_name', 'sensor_id', 'sensor_type', 'sensor_value', 'sensor_date')

    search_fields = ['sensor_name']


admin.site.register(Device, DeviceAdmin)
admin.site.register(Sensor, SensorAdmin)
