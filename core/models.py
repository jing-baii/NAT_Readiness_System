from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import FileExtensionValidator
import os
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.exceptions import ValidationError

def question_file_path(instance, filename):
    # Get the file extension
    ext = filename.split('.')[-1]
    # Create a new filename with timestamp
    new_filename = f"{instance.id}_{int(time.time())}.{ext}"
    return os.path.join('question_files', new_filename)

def avatar_path(instance, filename):
    # Save avatars in static/avatars directory
    return f'avatars/{instance.user.username}_{filename}'

class Topic(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class GeneralTopic(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    subject = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='general_topics')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['name', 'subject']

    def __str__(self):
        return f"{self.name} ({self.subject.name})"

class Subtopic(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    general_topic = models.ForeignKey(GeneralTopic, on_delete=models.CASCADE, related_name='subtopics')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['name', 'general_topic']

    def __str__(self):
        return f"{self.name} ({self.general_topic.name})"

class SchoolYear(models.Model):
    name = models.CharField(max_length=20, unique=True)
    start_date = models.DateField()
    end_date = models.DateField()

    def __str__(self):
        return self.name

class Question(models.Model):
    QUESTION_TYPES = [
        ('multiple_choice', 'Multiple Choice'),
        ('true_false', 'True/False'),
        ('short_answer', 'Short Answer'),
        ('essay', 'Essay'),
        ('file_upload', 'File Upload'),
    ]

    subtopic = models.ForeignKey(Subtopic, on_delete=models.CASCADE, related_name='questions')
    school_year = models.ForeignKey('SchoolYear', on_delete=models.CASCADE, related_name='questions', null=True, blank=True)
    question_text = models.TextField()
    question_type = models.CharField(max_length=20, choices=QUESTION_TYPES)
    correct_answer = models.TextField(blank=True, null=True)
    points = models.IntegerField(default=1)
    level = models.IntegerField(default=1)  # Level 1 for manually added questions, higher for generated ones
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    allow_file_upload = models.BooleanField(default=False)
    max_file_size = models.IntegerField(default=5, help_text="Maximum file size in MB")
    allowed_file_types = models.CharField(
        max_length=100,
        default='pdf,doc,docx,txt,jpg,jpeg,png,gif',
        help_text="Comma-separated list of allowed file extensions"
    )

    def __str__(self):
        return f"{self.subtopic.name} - {self.question_text[:50]}"

class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='choices')
    choice_text = models.CharField(max_length=200)
    is_correct = models.BooleanField(default=False)

    def __str__(self):
        return self.choice_text

class StudentResponse(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    answer = models.TextField()
    is_correct = models.BooleanField(default=False)
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.username} - {self.question.question_text[:50]}"

class QuizProgress(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    subtopic = models.ForeignKey(Subtopic, on_delete=models.CASCADE)
    time_elapsed = models.IntegerField(default=0)  # in seconds
    answers = models.JSONField(default=dict)  # stores question_id: answer pairs
    marked_for_review = models.JSONField(default=list)  # stores question_ids
    last_modified = models.DateTimeField(auto_now=True)
    is_completed = models.BooleanField(default=False)

    class Meta:
        unique_together = ['student', 'subtopic']
        verbose_name_plural = 'Quiz Progress'

    def __str__(self):
        return f"{self.student.username} - {self.subtopic.name}"

class StudyLink(models.Model):
    MATERIAL_TYPES = [
        ('video', 'Video'),
        ('text', 'Text'),
        ('interactive', 'Interactive'),
        ('quiz', 'Quiz'),
        ('practice', 'Practice')
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    url = models.URLField()
    subtopic = models.ForeignKey(Subtopic, on_delete=models.CASCADE)
    material_type = models.CharField(max_length=20, choices=MATERIAL_TYPES)
    source = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class LinkAccess(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    study_link = models.ForeignKey(StudyLink, on_delete=models.CASCADE)
    access_time = models.DateTimeField(auto_now_add=True)
    duration = models.DurationField(null=True, blank=True)

    def __str__(self):
        return f"{self.student.username} - {self.study_link.title}"

class FileUpload(models.Model):
    file = models.FileField(
        upload_to=question_file_path,
        validators=[
            FileExtensionValidator(
                allowed_extensions=['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'gif']
            )
        ]
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.ForeignKey('Question', on_delete=models.CASCADE, related_name='attachments')
    file_type = models.CharField(max_length=10)  # pdf, doc, txt, img

    def __str__(self):
        return f"{self.file.name} - {self.uploaded_at}"

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    avatar = models.ImageField(upload_to=avatar_path, null=True, blank=True)
    middle_initial = models.CharField(max_length=1, blank=True)
    name_extension = models.CharField(max_length=10, blank=True, help_text="E.g., Jr., Sr., III, etc.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        full_name = f"{self.user.first_name}"
        if self.middle_initial:
            full_name += f" {self.middle_initial}."
        full_name += f" {self.user.last_name}"
        if self.name_extension:
            full_name += f" {self.name_extension}"
        return full_name

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    try:
        instance.profile.save()
    except Profile.DoesNotExist:
        # If profile doesn't exist, create it
        Profile.objects.create(user=instance)

class SubjectQuizLevel(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    subject = models.ForeignKey(Topic, on_delete=models.CASCADE)
    level = models.IntegerField(default=1)
    last_attempt_date = models.DateTimeField(auto_now=True)
    is_completed = models.BooleanField(default=False)
    weak_areas = models.JSONField(default=dict)  # stores weak general_topics and subtopics
    total_attempts = models.IntegerField(default=0)
    highest_score = models.FloatField(default=0.0)

    class Meta:
        unique_together = ['student', 'subject', 'level']
        verbose_name_plural = 'Subject Quiz Levels'

    def __str__(self):
        return f"{self.student.username} - {self.subject.name} - Level {self.level}"

class Level(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Subject(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class QuestionGenerationSetting(models.Model):
    level = models.ForeignKey(Level, on_delete=models.CASCADE, related_name='generation_settings')
    questions_per_topic = models.PositiveIntegerField(default=5)
    easy_percentage = models.PositiveIntegerField(default=30)
    medium_percentage = models.PositiveIntegerField(default=50)
    hard_percentage = models.PositiveIntegerField(default=20)
    question_types = models.JSONField(default=list)
    passing_score = models.PositiveIntegerField(default=70, help_text='Minimum score required to pass this level (percentage)')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['level']
        verbose_name = 'Question Generation Setting'
        verbose_name_plural = 'Question Generation Settings'

    def clean(self):
        if self.easy_percentage + self.medium_percentage + self.hard_percentage != 100:
            raise ValidationError("Difficulty percentages must sum to 100%")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Settings for {self.level.name}"

class SurveyResponse(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Experience Ratings
    experience_rating = models.IntegerField()
    difficulty_rating = models.IntegerField()
    content_quality_rating = models.IntegerField()
    system_usability_rating = models.IntegerField()
    peu_control = models.IntegerField(default=0)
    
    # Feature Ratings
    quiz_quality_rating = models.IntegerField()
    study_materials_rating = models.IntegerField()
    progress_tracking_rating = models.IntegerField()
    recommendation_quality_rating = models.IntegerField()
    
    # Feature Usage
    helpful_features = models.JSONField()  # Store list of selected features
    most_used_features = models.JSONField()  # Store list of most used features
    least_used_features = models.JSONField()  # Store list of least used features
    
    # Learning Impact
    knowledge_improvement = models.IntegerField()
    confidence_improvement = models.IntegerField()
    study_habits_improvement = models.IntegerField()
    
    # Feedback
    suggestions = models.TextField()
    favorite_aspects = models.TextField()
    challenges_faced = models.TextField()
    additional_comments = models.TextField(blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    time_spent_on_system = models.IntegerField(default=0)  # in hours
    subjects_covered = models.JSONField(default=list)  # List of subjects completed

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Survey Response from {self.student.username} on {self.created_at}"

class UserLoginHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='login_history')
    timestamp = models.DateTimeField(auto_now_add=True)
    logout_time = models.DateTimeField(null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    session_duration = models.DurationField(null=True, blank=True)

    class Meta:
        verbose_name_plural = "User Login Histories"
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username} - {self.timestamp}"

    def record_logout(self):
        """Record the logout time and calculate session duration."""
        if not self.logout_time:
            self.logout_time = timezone.now()
            if self.timestamp:
                self.session_duration = self.logout_time - self.timestamp
            self.save()

class PerformanceMetric(models.Model):
    name = models.CharField(max_length=100, unique=True)
    numerator = models.IntegerField()
    denominator = models.IntegerField()
    value = models.FloatField(help_text="Raw value (e.g., 0.92 for 92%)")
    percentage = models.FloatField(help_text="Percentage value (e.g., 92 for 92%)")
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name}: {self.percentage}% ({self.numerator}/{self.denominator})"