# Generated by Django 5.1.7 on 2025-05-14 16:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0031_userloginhistory'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='userloginhistory',
            options={'ordering': ['-timestamp'], 'verbose_name_plural': 'User Login Histories'},
        ),
        migrations.AddField(
            model_name='userloginhistory',
            name='logout_time',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='userloginhistory',
            name='session_duration',
            field=models.DurationField(blank=True, null=True),
        ),
    ]
