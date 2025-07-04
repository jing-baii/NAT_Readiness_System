# Generated by Django 5.1.7 on 2025-05-01 06:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0012_recommendation'),
    ]

    operations = [
        migrations.AlterField(
            model_name='topic',
            name='name',
            field=models.CharField(max_length=100, unique=True),
        ),
        migrations.AlterUniqueTogether(
            name='generaltopic',
            unique_together={('name', 'subject')},
        ),
        migrations.AlterUniqueTogether(
            name='subtopic',
            unique_together={('name', 'general_topic')},
        ),
        migrations.DeleteModel(
            name='Recommendation',
        ),
    ]
