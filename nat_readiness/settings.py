import os
from pathlib import Path
import os

STCFI_NAT_Readiness = os.environ.get('STCFI_NAT_Readiness', '127.0.0.1')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables will not be loaded from .env file.")

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-your-secret-key-here')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'


ALLOWED_HOSTS = [STCFI_NAT_Readiness, 'localhost', '127.0.0.1', '192.168.43.176', 'testserver']

CRISPY_TEMPLATE_PACK = 'bootstrap4'  # For Bootstrap 4

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'core',
    'crispy_forms',
    'crispy_bootstrap4',
    'debug_toolbar',
    'mathfilters',
    'widget_tweaks',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]

ROOT_URLCONF = 'nat_readiness.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'core', 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'core.context_processors.achievement_status',
            ],
        },
    },
]

WSGI_APPLICATION = 'nat_readiness.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'core', 'static'),
]

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Login/Logout URLs
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'student_dashboard'
LOGOUT_REDIRECT_URL = 'login'

# Media files (user uploaded content)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
FILE_UPLOAD_PERMISSIONS = 0o644

# Create necessary directories
os.makedirs(os.path.join(BASE_DIR, 'core', 'static', 'avatars'), exist_ok=True)
os.makedirs(MEDIA_ROOT, exist_ok=True)

# Debug Toolbar settings
if DEBUG:
    INTERNAL_IPS = [
        '127.0.0.1',
    ]

# Educational Content API Settings
KHAN_ACADEMY_API_KEY = os.environ.get('KHAN_ACADEMY_API_KEY', '')
COURSERA_API_KEY = os.environ.get('COURSERA_API_KEY', '')

# OpenAI API Settings
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Llama API Settings
LLAMA_API_KEY = os.environ.get('LLAMA_API_KEY', '')

# Educational Content Service API Settings
EDUCATIONAL_API_KEY = os.environ.get('EDUCATIONAL_API_KEY', '')
EDUCATIONAL_API_URL = os.environ.get('EDUCATIONAL_API_URL', 'https://api.educational-content.com/v1')

# Cache settings
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': os.path.join(BASE_DIR, 'django_cache'),
        'TIMEOUT': 3600,  # 1 hour
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        }
    }
}

# Redis settings (for future use)
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/1')

# Rate limiting settings
RATE_LIMIT_ENABLED = True
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds

# Educational Content Service Settings
EDUCATIONAL_CONTENT = {
    'CACHE_TIMEOUT': 3600,  # 1 hour
    'RATE_LIMIT': 100,  # requests per minute
    'MODEL_PATH': os.path.join(BASE_DIR, 'models'),
    'RELIABLE_SOURCES': {
        'Google Scholar': {
            'url': 'https://scholar.google.com',
            'search_url': 'https://scholar.google.com/scholar',
            'reliability_score': 0.9,
            'reliable_channels': []
        },
        'YouTube': {
            'url': 'https://www.youtube.com',
            'search_url': 'https://www.youtube.com/results',
            'reliability_score': 0.8,
            'reliable_channels': [
                'Khan Academy',
                'CrashCourse',
                'MIT OpenCourseWare',
                'Stanford Online'
            ]
        },
        'Semantic Scholar': {
            'url': 'https://www.semanticscholar.org',
            'search_url': 'https://api.semanticscholar.org/graph/v1/paper/search',
            'reliability_score': 0.95,
            'reliable_channels': []
        }
    }
}

# Create models directory if it doesn't exist
os.makedirs(EDUCATIONAL_CONTENT['MODEL_PATH'], exist_ok=True)

