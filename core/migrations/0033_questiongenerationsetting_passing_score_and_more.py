# Generated by Django 5.1.7 on 2025-05-18 22:10

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0032_alter_userloginhistory_options_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='questiongenerationsetting',
            name='passing_score',
            field=models.PositiveIntegerField(default=70, help_text='Minimum score required to pass this level (percentage)'),
        ),
        migrations.CreateModel(
            name='SurveyResponse',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('experience_rating', models.IntegerField()),
                ('difficulty_rating', models.IntegerField()),
                ('content_quality_rating', models.IntegerField()),
                ('system_usability_rating', models.IntegerField()),
                ('quiz_quality_rating', models.IntegerField()),
                ('study_materials_rating', models.IntegerField()),
                ('progress_tracking_rating', models.IntegerField()),
                ('recommendation_quality_rating', models.IntegerField()),
                ('helpful_features', models.JSONField()),
                ('most_used_features', models.JSONField()),
                ('least_used_features', models.JSONField()),
                ('knowledge_improvement', models.IntegerField()),
                ('confidence_improvement', models.IntegerField()),
                ('study_habits_improvement', models.IntegerField()),
                ('suggestions', models.TextField()),
                ('favorite_aspects', models.TextField()),
                ('challenges_faced', models.TextField()),
                ('additional_comments', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('time_spent_on_system', models.IntegerField(default=0)),
                ('subjects_covered', models.JSONField(default=list)),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
    ]
