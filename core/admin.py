from django.contrib import admin
from django.utils.html import format_html
from django.urls import path
from django.shortcuts import render
from django.db.models import Count, Prefetch
from .models import Topic, GeneralTopic, Subtopic, Question, StudentResponse, StudyLink, LinkAccess, FileUpload, Choice, User, SchoolYear, PerformanceMetric

@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    search_fields = ('name', 'description')

@admin.register(GeneralTopic)
class GeneralTopicAdmin(admin.ModelAdmin):
    list_display = ('name', 'subject', 'created_at')
    list_filter = ('subject',)
    search_fields = ('name', 'description')

@admin.register(Subtopic)
class SubtopicAdmin(admin.ModelAdmin):
    list_display = ('name', 'general_topic', 'created_at')
    list_filter = ('general_topic__subject', 'general_topic')
    search_fields = ('name', 'description')

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('question_text', 'subtopic', 'question_type', 'points', 'created_at')
    list_filter = ('subtopic__general_topic__subject', 'subtopic__general_topic', 'subtopic', 'question_type')
    search_fields = ('question_text', 'subtopic__name')

@admin.register(StudentResponse)
class StudentResponseAdmin(admin.ModelAdmin):
    list_display = ('student', 'question', 'is_correct', 'submitted_at')
    list_filter = ('is_correct', 'submitted_at', 'question__subtopic__general_topic__subject')
    search_fields = ('student__username', 'question__question_text')

@admin.register(StudyLink)
class StudyLinkAdmin(admin.ModelAdmin):
    list_display = ('title', 'subtopic', 'material_type', 'source', 'created_at', 'access_count')
    list_filter = ('material_type', 'subtopic__general_topic__subject', 'subtopic__general_topic', 'subtopic', 'created_at')
    search_fields = ('title', 'description', 'url', 'source')
    list_per_page = 20
    readonly_fields = ('created_at',)
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'description', 'url', 'material_type', 'source')
        }),
        ('Topic Information', {
            'fields': ('subtopic',)
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )

    def access_count(self, obj):
        return obj.linkaccess_set.count()
    access_count.short_description = 'Access Count'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('subtopic', 'subtopic__general_topic', 'subtopic__general_topic__subject')

class LinkAccessInline(admin.TabularInline):
    model = LinkAccess
    extra = 0
    readonly_fields = ('student', 'access_time', 'duration')
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False

StudyLinkAdmin.inlines = [LinkAccessInline]

@admin.register(FileUpload)
class FileUploadAdmin(admin.ModelAdmin):
    list_display = ('file', 'question', 'uploaded_by', 'uploaded_at')
    list_filter = ('uploaded_at', 'file_type')
    search_fields = ('file', 'question__question_text')

@admin.register(Choice)
class ChoiceAdmin(admin.ModelAdmin):
    list_display = ('choice_text', 'question', 'is_correct')
    list_filter = ('is_correct', 'question__subtopic__general_topic__subject')
    search_fields = ('choice_text', 'question__question_text')

@admin.register(LinkAccess)
class LinkAccessAdmin(admin.ModelAdmin):
    list_display = ('student', 'study_link', 'access_time', 'duration', 'subject_name')
    list_filter = ('access_time', 'study_link__subtopic__general_topic__subject', 'student')
    search_fields = ('student__username', 'study_link__title', 'study_link__subtopic__name')
    ordering = ('-access_time',)
    list_per_page = 50
    actions = ['delete_selected']

    def subject_name(self, obj):
        return obj.study_link.subtopic.general_topic.subject.name
    subject_name.short_description = 'Subject'

    def has_delete_permission(self, request, obj=None):
        return request.user.is_staff

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('student-access-report/', self.admin_site.admin_view(self.student_access_report), name='student-access-report'),
        ]
        return custom_urls + urls

    def student_access_report(self, request):
        # Get all students with their accessed study links
        students = User.objects.filter(
            is_staff=False,
            linkaccess__isnull=False
        ).distinct().prefetch_related(
            Prefetch(
                'linkaccess_set',
                queryset=LinkAccess.objects.select_related(
                    'study_link',
                    'study_link__subtopic',
                    'study_link__subtopic__general_topic',
                    'study_link__subtopic__general_topic__subject'
                ).order_by('-access_time')
            )
        )

        # Group study links by subject for each student
        student_data = []
        for student in students:
            student_links = student.linkaccess_set.all()
            subjects = {}
            
            for access in student_links:
                subject = access.study_link.subtopic.general_topic.subject
                if subject not in subjects:
                    subjects[subject] = []
                subjects[subject].append(access)
            
            student_data.append({
                'student': student,
                'subjects': subjects,
                'total_accesses': student_links.count()
            })

        context = {
            'title': 'Student Study Link Access Report',
            'student_data': student_data,
            'opts': self.model._meta,
            'has_view_permission': True,
        }
        return render(request, 'admin/student_access_report.html', context)

# Add the report link to the admin index
admin.site.index_template = 'admin/custom_index.html'

admin.site.register(SchoolYear)

@admin.register(PerformanceMetric)
class PerformanceMetricAdmin(admin.ModelAdmin):
    list_display = ('name', 'numerator', 'denominator', 'value', 'percentage', 'created_at', 'updated_at')
    search_fields = ('name',)
    readonly_fields = ('created_at', 'updated_at')