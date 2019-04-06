# Generated by Django 2.1.7 on 2019-04-01 15:10

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('devman', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='device',
            name='device_date',
            field=models.DateTimeField(default=django.utils.timezone.now, verbose_name='device created'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='sensor',
            name='sensor_date',
            field=models.DateTimeField(default=django.utils.timezone.now, verbose_name='sensor created'),
            preserve_default=False,
        ),
    ]
