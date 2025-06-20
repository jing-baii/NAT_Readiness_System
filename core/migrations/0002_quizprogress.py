# Generated by Django 5.1.7 on 2025-04-03 14:17

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='QuizProgress',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('time_elapsed', models.IntegerField(default=0)),
                ('answers', models.JSONField(default=dict)),
                ('marked_for_review', models.JSONField(default=list)),
                ('last_modified', models.DateTimeField(auto_now=True)),
                ('is_completed', models.BooleanField(default=False)),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('subtopic', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.subtopic')),
            ],
            options={
                'verbose_name_plural': 'Quiz Progress',
                'unique_together': {('student', 'subtopic')},
            },
        ),
    ]
