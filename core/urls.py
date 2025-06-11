from django.urls import path
from . import views
from django.shortcuts import redirect
from .views import api_get_correct_answer

def root_redirect(request):
    if request.user.is_authenticated:
        return redirect('student_dashboard')
    return redirect('login')

urlpatterns = [
    # Root URL
    path('', root_redirect, name='root'),
    
    # Student URLs
    path('student_dashboard/', views.student_dashboard, name='student_dashboard'),
    path('register/', views.register, name='register'),
    path('submit-answer/', views.submit_answer, name='submit_answer'),
    path('get_recommendations/', views.get_recommendations, name='get_recommendations'),
    path('track-link/<int:link_id>/', views.track_link_access, name='track_link_access'),
    path('profile/update/', views.update_profile, name='update_profile'),
    path('student/<int:student_id>/', views.student_details, name='student_details'),
    path('student/<int:student_id>/change-credentials/', views.change_student_credentials, name='change_student_credentials'),
    path('change-password/', views.change_password, name='change_password'),
    path('generate-study-materials/', views.generate_study_materials, name='generate_study_materials'),
    path('help/', views.help_view, name='help'),
    path('privacy-policy/', views.privacy_policy, name='privacy_policy'),
    path('terms/', views.terms, name='terms'),
    path('contact/', views.contact, name='contact'),

    # Admin URLs
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('questions/', views.list_questions, name='list_questions'),
    path('questions/add/', views.add_question, name='add_question'),
    path('questions/<int:question_id>/edit/', views.edit_question, name='edit_question'),
    path('questions/<int:question_id>/delete/', views.delete_question, name='delete_question'),
    path('study-links/add/', views.add_study_link, name='add_study_link'),
    path('student-access-report/', views.student_access_report, name='student_access_report'),
    path('admin/change-credentials/', views.change_admin_credentials, name='change_admin_credentials'),
    
    # Subject Management URLs
    path('subject/add/', views.add_subject, name='add_subject'),
    path('subject/<int:subject_id>/general-topics/', views.get_general_topics, name='get_general_topics'),
    
    # General Topic Management URLs
    path('general-topic/add/', views.add_general_topic, name='add_general_topic'),
    path('general-topic/<int:general_topic_id>/subtopics/', views.get_subtopics, name='get_subtopics'),
    
    # Link Access Management
    path('link-access/<int:access_id>/delete/', views.delete_link_access, name='delete_link_access'),
    
    # Subtopic Management URLs
    path('subtopic/add/', views.add_subtopic, name='add_subtopic'),
    path('save-quiz-progress/', views.save_quiz_progress, name='save_quiz_progress'),
    path('start-practice/<int:subtopic_id>/', views.start_practice, name='start_practice'),
    path('list-responses/', views.list_responses, name='list_responses'),
    path('dashboard/student-performance/', views.student_performance, name='student_performance'),
    path('question-stats/', views.question_stats, name='question_stats'),
    path('question-details/<int:question_id>/', views.question_details, name='question_details'),
    path('view-response/<int:response_id>/', views.view_response, name='view_response'),
    path('subject-performance/<str:subject_name>/', views.subject_performance_details, name='subject_performance_details'),
    path('view-all-responses/', views.view_all_responses, name='view_all_responses'),
    path('response-details/<int:response_id>/', views.view_response_details, name='view_response_details'),
    path('take-quiz/<int:subject_id>/', views.take_subject_quiz, name='take_subject_quiz'),
    path('api/performance/<int:subject_id>/', views.get_performance_data, name='get_performance_data'),
    path('api/study-materials/<int:subject_id>/', views.get_study_materials, name='get_study_materials'),
    path('manage-quiz-attempts/', views.manage_quiz_attempts, name='manage_quiz_attempts'),
    
    # Quiz URLs - Order is important here
    path('quiz/level/', views.quiz_by_level, name='quiz_by_level'),  # Subject selection by level
    path('quiz/', views.quiz, name='quiz'),  # Main quiz page with subject selection
    path('quiz/<int:subject_id>/', views.take_subject_quiz, name='take_subject_quiz'),  # Subject-specific quiz page
    path('quiz/<int:subject_id>/level/<int:level_number>/', views.take_level_quiz, name='take_level_quiz'),
    path('quiz/<int:subject_id>/submit/', views.submit_quiz, name='submit_quiz'),
    path('quiz/<int:subject_id>/levels/', views.get_subject_levels, name='get_subject_levels'),
    path('quiz/<int:subtopic_id>/', views.take_quiz, name='take_quiz'),
    
    # Study Material URLs
    path('study-material-preferences/<int:subject_id>/<str:score>/', views.study_material_preferences, name='study_material_preferences'),
    path('subjects/', views.get_all_subjects, name='get_all_subjects'),
    path('questions/bulk_add/', views.bulk_add_questions, name='bulk_add_questions'),

    # School Year Management URL
    path('school-year/add/', views.add_school_year, name='add_school_year'),

    # New URL pattern
    path('extract-question/', views.extract_question, name='extract_question'),

    path('check-general-topic-exists/', views.check_general_topic_exists, name='check_general_topic_exists'),
    path('check-subtopic-exists/', views.check_subtopic_exists, name='check_subtopic_exists'),
    path('suggest-topics/', views.suggest_topics, name='suggest_topics'),

    # Question Generation Settings
    path('question-generation-settings/', views.question_generation_settings, name='question_generation_settings'),
    path('api/settings/<int:setting_id>/', views.get_setting, name='get_setting'),
    path('api/settings/<int:setting_id>/update/', views.update_level_setting, name='update_level_setting'),
    path('api/settings/<int:setting_id>/delete/', views.delete_level_setting, name='delete_level_setting'),
    path('api/level/<int:level_id>/settings/', views.get_level_settings, name='get_level_settings'),
    path('api/level/<int:level_id>/settings/create/', views.create_level_setting, name='create_level_setting'),
    path('api/level/create/', views.create_level, name='create_level'),
    path('api/levels/', views.get_all_levels, name='get_all_levels'),
    path('congratulations/', views.congratulations, name='congratulations'),
    path('survey/', views.survey, name='survey'),
    path('api/get-correct-answer/', api_get_correct_answer, name='api_get_correct_answer'),
    path('analytics/', views.view_analytics, name='view_analytics'),
    path('api/analytics/performance-data/', views.get_performance_analytics_data, name='get_performance_analytics_data'),
    path('api/analytics/debug/', views.debug_analytics_data, name='debug_analytics_data'),
    path('api/analytics/activity-logs/', views.get_activity_logs_api, name='activity_logs_api'),
    path('api/analytics/survey-data/', views.get_survey_data, name='get_survey_data'),
    path('subject-hierarchy/', views.subject_hierarchy, name='subject_hierarchy'),
    path('api/subject-hierarchy/', views.get_subject_hierarchy, name='get_subject_hierarchy'),
    path('display-hierarchy/', views.display_hierarchy, name='display_hierarchy'),
    path('download-hierarchy/', views.download_hierarchy_docx, name='download_hierarchy_docx'),
    path('get-performance-analytics-data/', views.get_performance_analytics_data, name='get_performance_analytics_data'),
    path('student-weak-topics/<int:student_id>/', views.student_weak_topics_ajax, name='student_weak_topics_ajax'),
    path('api/question-count/', views.question_count, name='question_count'),
    path('api/generate-questions/', views.generate_questions, name='generate_questions'),
    path('api/add-manual-question/', views.add_manual_question, name='add_manual_question'),
    path('api/get-subtopic-data/<int:subtopic_id>/', views.get_subtopic_data, name='get_subtopic_data'),
    path('api/school-years/', views.get_school_years, name='get_school_years'),
    path('performance-metrics/', views.performance_metrics_view, name='performance_metrics'),
]