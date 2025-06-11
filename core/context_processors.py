from django.contrib.auth.models import User
from core.models import Topic, StudentResponse, SubjectQuizLevel
import logging

logger = logging.getLogger(__name__)

def achievement_status(request):
    """
    Context processor to check if a student has completed all level 1 quizzes
    with a score of 90% or higher, or has eliminated all weaknesses.
    """
    if not request.user.is_authenticated or request.user.is_staff:
        return {
            'has_completed_level_1': False,
            'has_eliminated_weaknesses': False
        }

    try:
        # Get all subjects
        subjects = Topic.objects.all()
        logger.info(f"Found {subjects.count()} subjects")

        # Check if user has taken any quizzes at all
        has_taken_any_quiz = StudentResponse.objects.filter(student=request.user).exists()
        if not has_taken_any_quiz:
            logger.info("User has not taken any quizzes")
            return {
                'has_completed_level_1': False,
                'has_eliminated_weaknesses': False
            }

        # Check if user has taken at least one quiz for every subject
        for subject in subjects:
            has_taken_subject_quiz = StudentResponse.objects.filter(
                student=request.user,
                question__subtopic__general_topic__subject=subject
            ).exists()
            if not has_taken_subject_quiz:
                logger.info(f"User has not taken any quizzes for subject: {subject.name}")
                return {
                    'has_completed_level_1': False,
                    'has_eliminated_weaknesses': False
                }

        # Check if all level 1 quizzes are completed with 90% or higher
        all_level_1_completed = True
        for subject in subjects:
            responses = StudentResponse.objects.filter(
                student=request.user,
                question__subtopic__general_topic__subject=subject,
                question__level=1
            )
            total_questions = responses.count()
            if total_questions == 0:
                all_level_1_completed = False
                logger.info(f"No level 1 responses found for subject: {subject.name}")
                break
            correct_answers = responses.filter(is_correct=True).count()
            score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            if score < 90:
                all_level_1_completed = False
                logger.info(f"Level 1 score for {subject.name} is {score}% (needs 90%)")
                break

        # Check if all weaknesses are eliminated
        all_weaknesses_eliminated = True
        for subject in subjects:
            quiz_level = SubjectQuizLevel.objects.filter(
                student=request.user,
                subject=subject
            ).first()
            if quiz_level and quiz_level.weak_areas:
                weak_general_topics = quiz_level.weak_areas.get('general_topics', [])
                weak_subtopics = quiz_level.weak_areas.get('subtopics', [])
                if weak_general_topics or weak_subtopics:
                    all_weaknesses_eliminated = False
                    logger.info(f"Found weaknesses in subject: {subject.name}")
                    break

        logger.info(f"Final status - Level 1 completed: {all_level_1_completed}, Weaknesses eliminated: {all_weaknesses_eliminated}")
        return {
            'has_completed_level_1': all_level_1_completed,
            'has_eliminated_weaknesses': all_weaknesses_eliminated
        }
    except Exception as e:
        logger.error(f"Error in achievement_status: {str(e)}")
        return {
            'has_completed_level_1': False,
            'has_eliminated_weaknesses': False
        }
 