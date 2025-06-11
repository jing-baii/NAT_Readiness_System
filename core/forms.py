from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Question, FileUpload, Choice, SchoolYear
import json


class StudentRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    middle_initial = forms.CharField(max_length=1, required=False)
    last_name = forms.CharField(max_length=30, required=True)
    name_extension = forms.CharField(max_length=10, required=False, help_text="E.g., Jr., Sr., III, etc.")

    class Meta:
        model = User
        fields = ('first_name', 'middle_initial', 'last_name', 'name_extension', 'username', 'email', 'password1', 'password2')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        
        # Combine last name with extension if provided
        last_name = self.cleaned_data['last_name']
        name_extension = self.cleaned_data.get('name_extension', '').strip()
        if name_extension:
            last_name = f"{last_name} {name_extension}"
        user.last_name = last_name
        
        # Store middle initial in the database (you might need to create a custom user model or profile)
        if commit:
            user.save()
            
            # Create or update user profile with middle initial
            from .models import Profile
            profile, created = Profile.objects.get_or_create(user=user)
            profile.middle_initial = self.cleaned_data.get('middle_initial', '')
            profile.name_extension = name_extension
            profile.save()
            
        return user

class QuestionForm(forms.ModelForm):
    # Multiple choice fields
    choices = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="JSON string containing choices data"
    )

    # True/False field
    true_false_answer = forms.ChoiceField(
        choices=[('True', 'True'), ('False', 'False')],
        required=False,
        widget=forms.RadioSelect(),
        label='Correct Answer'
    )

    # Essay/Short Answer Fields
    essay_answer = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 4, 'class': 'form-control'}),
        help_text="Enter the correct answer"
    )

    # File Upload Fields
    file1 = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.gif'
        }),
        help_text="Upload a file (PDF, DOC, TXT, Images). Maximum size: 5MB."
    )
    
    file2 = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.gif'
        }),
        help_text="Upload another file (optional)"
    )
    
    file3 = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.gif'
        }),
        help_text="Upload another file (optional)"
    )

    # Add school year field
    school_year = forms.ModelChoiceField(
        queryset=None,
        required=True,
        label='School Year',
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    class Meta:
        model = Question
        fields = ['subtopic', 'school_year', 'question_text', 'question_type', 'points', 
                 'max_file_size', 'allowed_file_types', 'level']
        widgets = {
            'question_text': forms.Textarea(attrs={
                'rows': 4,
                'class': 'form-control',
                'placeholder': 'Enter your question here...'
            }),
            'subtopic': forms.Select(attrs={'class': 'form-select'}),
            'school_year': forms.Select(attrs={'class': 'form-select'}),
            'question_type': forms.Select(attrs={'class': 'form-select'}),
            'points': forms.NumberInput(attrs={'class': 'form-control'}),
            'max_file_size': forms.NumberInput(attrs={'class': 'form-control'}),
            'allowed_file_types': forms.TextInput(attrs={'class': 'form-control'}),
            'level': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1',
                'placeholder': 'Enter question level'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['school_year'].queryset = SchoolYear.objects.all().order_by('-start_date')
        self.fields['level'].required = True
        self.fields['level'].min_value = 1

    def clean(self):
        cleaned_data = super().clean()
        question_type = cleaned_data.get('question_type')
        
        if question_type == 'multiple_choice':
            try:
                choices_data = json.loads(cleaned_data.get('choices', '[]'))
                if not choices_data:
                    raise forms.ValidationError('Multiple choice questions must have at least one choice.')
                
                # Validate that there's exactly one correct choice
                correct_choices = [choice for choice in choices_data if choice.get('is_correct')]
                if len(correct_choices) != 1:
                    raise forms.ValidationError('Please select exactly one correct choice.')
                
                # Validate choice text
                for choice in choices_data:
                    if not choice.get('text', '').strip():
                        raise forms.ValidationError('Choice text cannot be empty.')
                
            except json.JSONDecodeError:
                raise forms.ValidationError('Invalid choices data format.')
        
        elif question_type == 'true_false':
            if not cleaned_data.get('true_false_answer'):
                raise forms.ValidationError('Please select the correct answer (True/False).')
        
        elif question_type in ['short_answer', 'essay']:
            if not cleaned_data.get('essay_answer'):
                raise forms.ValidationError('Please provide the correct answer.')
        
        elif question_type == 'file_upload':
            if not cleaned_data.get('max_file_size'):
                raise forms.ValidationError('Please specify maximum file size for file uploads.')
            if not cleaned_data.get('allowed_file_types'):
                raise forms.ValidationError('Please specify allowed file types for file uploads.')
        
        return cleaned_data

    def clean_level(self):
        level = self.cleaned_data.get('level')
        if level is None:
            raise forms.ValidationError('Level is required')
        try:
            level = int(level)
            if level < 1:
                raise forms.ValidationError('Level must be at least 1')
            return level
        except (ValueError, TypeError):
            raise forms.ValidationError('Level must be a valid number')