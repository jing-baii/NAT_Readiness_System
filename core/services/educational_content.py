import requests
from django.core.cache import cache
from django.conf import settings
from django.utils import timezone
from ..models import StudyLink, Question, GeneralTopic, Subtopic
import logging
from typing import List, Dict, Optional
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os
from urllib.parse import quote, quote_plus
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry
import json
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable symlinks warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

logger = logging.getLogger(__name__)

class EducationalContentService:
    def __init__(self):
        self.cache = {}
        self.sources = settings.EDUCATIONAL_CONTENT['RELIABLE_SOURCES']
        self.cache_timeout = settings.EDUCATIONAL_CONTENT['CACHE_TIMEOUT']
        self.rate_limit = settings.EDUCATIONAL_CONTENT['RATE_LIMIT']
        self.rate_limit_window = 60  # seconds
        self.last_request_time = 0
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.model_path = settings.EDUCATIONAL_CONTENT['MODEL_PATH']
        self._load_or_train_model()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.inappropriate_keywords = [
            'adult', 'porn', 'xxx', 'sex', 'nude', 'naked', 'explicit',
            'violence', 'gore', 'hate', 'racism', 'terrorism', 'drugs'
        ]
        
        # Initialize Llama model and tokenizer with a smaller model
        self._initialize_model()

    def _initialize_model(self):
        try:
            # First try to initialize with GPU support
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                quantization_config=quantization_config,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            
        except Exception as e:
            logger.warning(f"GPU initialization failed: {str(e)}. Falling back to CPU mode.")
            try:
                # Fallback to CPU mode
                self.model = AutoModelForCausalLM.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    device_map="cpu"
                )
                self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            except Exception as e:
                logger.error(f"Failed to initialize model in CPU mode: {str(e)}")
                raise

    def _load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            # Try to load existing model
            self.vectorizer = joblib.load(os.path.join(self.model_path, 'vectorizer.joblib'))
            self.knn = joblib.load(os.path.join(self.model_path, 'knn.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.joblib'))
            logger.info("Loaded existing KNN model")
        except:
            # Train new model if none exists
            self._train_model()
            logger.info("Trained new KNN model")

    def _train_model(self):
        """Train the KNN model with existing questions"""
        try:
            # Get all questions
            questions = Question.objects.all()
            
            # Create feature vectors
            texts = [f"{q.question_text} {q.subtopic.name} {q.subtopic.general_topic.name}" 
                    for q in questions]
            
            # Fit and transform texts
            X = self.vectorizer.fit_transform(texts)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X.toarray())
            
            # Fit KNN model
            self.knn.fit(X_scaled)
            
            # Save models
            os.makedirs(self.model_path, exist_ok=True)
            joblib.dump(self.vectorizer, os.path.join(self.model_path, 'vectorizer.joblib'))
            joblib.dump(self.knn, os.path.join(self.model_path, 'knn.joblib'))
            joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.joblib'))
            
            logger.info("Successfully trained and saved KNN model")
        except Exception as e:
            logger.error(f"Error training KNN model: {str(e)}")
            raise

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        if current_time - self.last_request_time < (1.0 / (self.rate_limit / self.rate_limit_window)):
            time.sleep((1.0 / (self.rate_limit / self.rate_limit_window)) - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    def _get_cached_content(self, key: str) -> Optional[List[Dict]]:
        """Get cached content"""
        return self.cache.get(key)

    def _set_cached_content(self, key: str, content: List[Dict]) -> None:
        """Set cached content"""
        self.cache[key] = content

    def _classify_weak_subtopic(self, question: Question) -> str:
        """Classify the weak subtopic using KNN"""
        try:
            # Get all questions and their subtopics
            questions = Question.objects.select_related('subtopic').all()
            
            # Create feature vectors
            texts = [f"{q.question_text} {q.subtopic.name}" for q in questions]
            subtopics = [q.subtopic.name for q in questions]
            
            # Fit and transform texts
            X = self.vectorizer.fit_transform(texts)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X.toarray())
            
            # Fit KNN model
            self.knn.fit(X_scaled)
            
            # Get the current question's features
            current_text = f"{question.question_text} {question.subtopic.name}"
            current_features = self.vectorizer.transform([current_text])
            current_features_scaled = self.scaler.transform(current_features.toarray())
            
            # Find nearest neighbors
            distances, indices = self.knn.kneighbors(current_features_scaled)
            
            # Get the most common subtopic among neighbors
            neighbor_subtopics = [subtopics[i] for i in indices[0]]
            weak_subtopic = max(set(neighbor_subtopics), key=neighbor_subtopics.count)
            
            return weak_subtopic
        except Exception as e:
            logger.error(f"Error in _classify_weak_subtopic: {str(e)}")
            return question.subtopic.name

    def _search_google_scholar(self, query: str) -> List[Dict]:
        """
        Search Google Scholar with improved error handling and content parsing.
        """
        try:
            search_url = f"{self.sources['Google Scholar']['search_url']}?q={quote_plus(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for item in soup.select('.gs_r.gs_or.gs_scl'):
                try:
                    title_elem = item.select_one('.gs_rt a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    url = title_elem['href']
                    
                    # Skip PDF links
                    if url.endswith('.pdf'):
                        continue
                    
                    description = ''
                    desc_elem = item.select_one('.gs_rs')
                    if desc_elem:
                        description = desc_elem.text.strip()
                    
                    author = ''
                    author_elem = item.select_one('.gs_a')
                    if author_elem:
                        author = author_elem.text.strip()
                    
                    results.append({
                        'title': title,
                        'description': description,
                        'url': url,
                        'author': author,
                        'source': 'Google Scholar',
                        'type': 'article',
                        'reliability_score': self.sources['Google Scholar']['reliability_score']
                    })
                except Exception as e:
                    logger.warning(f"Error parsing Google Scholar result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")
            return []

    def _search_youtube(self, query: str) -> List[Dict]:
        """
        Search YouTube with improved error handling and content filtering.
        """
        try:
            search_url = f"{self.sources['YouTube']['search_url']}?search_query={quote_plus(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for item in soup.select('ytd-video-renderer'):
                try:
                    title_elem = item.select_one('#video-title')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    url = f"https://www.youtube.com{title_elem['href']}"
                    
                    channel_elem = item.select_one('#channel-name')
                    channel = channel_elem.text.strip() if channel_elem else ''
                    
                    # Check if channel is reliable
                    is_reliable = channel in self.sources['YouTube']['reliable_channels']
                    
                    description = ''
                    desc_elem = item.select_one('#description-text')
                    if desc_elem:
                        description = desc_elem.text.strip()
                    
                    results.append({
                        'title': title,
                        'description': description,
                        'url': url,
                        'author': channel,
                        'source': 'YouTube',
                        'type': 'video',
                        'reliability_score': self.sources['YouTube']['reliability_score'] if is_reliable else 0.5
                    })
                except Exception as e:
                    logger.warning(f"Error parsing YouTube result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching YouTube: {str(e)}")
            return []

    def _search_semantic_scholar(self, query: str) -> List[Dict]:
        """
        Search Semantic Scholar with improved error handling and content parsing.
        """
        try:
            search_url = f"{self.sources['Semantic Scholar']['search_url']}?query={quote_plus(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for paper in data.get('data', [])[:10]:
                try:
                    results.append({
                        'title': paper.get('title', ''),
                        'description': paper.get('abstract', ''),
                        'url': paper.get('url', ''),
                        'author': ', '.join(author.get('name', '') for author in paper.get('authors', [])),
                        'source': 'Semantic Scholar',
                        'type': 'article',
                        'citations': paper.get('citationCount', 0),
                        'reliability_score': self.sources['Semantic Scholar']['reliability_score']
                    })
                except Exception as e:
                    logger.warning(f"Error parsing Semantic Scholar result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            return []

    def _get_cache_key(self, query, topic_id):
        """Generate a safe cache key by hashing the query and topic_id"""
        # Clean the query by removing special characters and spaces
        clean_query = ''.join(c for c in query if c.isalnum() or c in ('-', '_'))
        # Create a safe key string
        key_string = f"ed_content_{clean_query}_{topic_id}"
        # Use MD5 to create a fixed-length, safe key
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    @sleep_and_retry
    @limits(calls=100, period=60)  # 100 calls per minute
    def search_educational_content(self, query: str, material_type: str = None) -> List[Dict]:
        """
        Search for educational content across multiple sources with rate limiting.
        """
        cache_key = self._get_cache_key(query, material_type or '')
        
        # Try to get from cache first
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            results = []
            
            # Search Google Scholar
            scholar_results = self._search_google_scholar(query)
            results.extend(scholar_results)
            
            # Search YouTube if material type is video
            if material_type == 'video':
                youtube_results = self._search_youtube(query)
                results.extend(youtube_results)
            
            # Search Semantic Scholar
            semantic_results = self._search_semantic_scholar(query)
            results.extend(semantic_results)
            
            # Filter inappropriate content
            results = self._filter_inappropriate_content(results)
            
            # Score and sort results
            results = self._score_and_sort_results(results)
            
            # Cache results
            cache.set(cache_key, results, self.cache_timeout)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching educational content: {str(e)}")
            return []

    def _generate_with_llama(self, prompt: str, max_length: int = 1000) -> str:
        """Generate text using Llama model"""
        if not self.model or not self.tokenizer:
            logger.error("Llama model not initialized")
            return ""
            
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response with more conservative parameters
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.5,  # Lower temperature for more focused responses
                do_sample=True,
                top_p=0.85,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2  # Add repetition penalty
            )
            
            # Decode the generated text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            response = response[len(prompt):].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error generating with Llama: {str(e)}")
            return ""

    def _generate_with_ollama(self, prompt: str, max_length: int = 1000) -> str:
        """Generate text using Ollama with Mistral model"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        import time
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,  # number of retries
            backoff_factor=1,  # wait 1, 2, 4 seconds between retries
            status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
        )
        
        # Create a session with retry strategy
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        try:
            # Ollama API endpoint
            url = "http://localhost:11434/api/generate"
            
            # Prepare the payload for Ollama
            payload = {
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": max_length,
                    "num_ctx": 4096,  # Increased context window
                    "num_thread": 4,   # Use 4 threads for processing
                    "num_gpu": 1       # Use 1 GPU if available
                }
            }
            
            # Make the API call with increased timeout
            response = session.post(url, json=payload, timeout=600)  # 10 minutes timeout
            response.raise_for_status()
            
            # Extract the generated text
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            logger.error("Ollama API request timed out after 10 minutes. The model might be taking too long to generate a response.")
            return ""
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama API. Please ensure the Ollama server is running.")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to Ollama API: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API call: {str(e)}")
            return ""

    def generate_personalized_questions(self, weak_areas: List[str], subject: str, level: int, num_questions: int = 5) -> List[Dict]:
        """
        Generate personalized questions based on student's weak areas using Ollama with Mistral.
        """
        cache_key = self._get_cache_key(f"personalized_questions_{subject}_{level}", '_'.join(weak_areas))
        
        # Try to get from cache first
        cached_questions = cache.get(cache_key)
        if cached_questions:
            logger.debug(f"Retrieved personalized questions from cache for subject: {subject}, level: {level}")
            return cached_questions

        try:
            # Prepare the prompt for Mistral
            prompt = f"""Generate {num_questions} multiple-choice questions for a {subject} quiz at level {level}.
            Focus on these weak areas: {', '.join(weak_areas)}.
            Each question should have:
            1. A clear question stem
            2. 4 answer choices (A, B, C, D)
            3. One correct answer
            4. Difficulty appropriate for level {level}
            
            Format each question as JSON with these fields:
            - question_text: The question stem
            - choices: List of 4 answer choices
            - correct_answer: The correct choice (A, B, C, or D)
            - explanation: Brief explanation of the correct answer
            - points: Difficulty points (1-5)
            
            Return only the JSON array of questions."""

            # Generate response using Ollama with Mistral
            response = self._generate_with_ollama(prompt)
            
            # Parse the response
            try:
                questions = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    questions = json.loads(json_match.group())
                else:
                    logger.error("Failed to parse Ollama response as JSON")
                    return []

            # Cache the results
            cache.set(cache_key, questions, self.cache_timeout)
            logger.debug(f"Cached personalized questions for subject: {subject}, level: {level}")
            
            return questions

        except Exception as e:
            logger.error(f"Error generating personalized questions: {str(e)}")
            return []

    def get_quiz_questions(self, topic: str, level: int, weak_areas: List[str] = None) -> List[Dict]:
        """
        Get quiz questions for a specific topic and level, with optional focus on weak areas.
        """
        if weak_areas:
            # Generate personalized questions for weak areas
            personalized_questions = self.generate_personalized_questions(weak_areas, topic, level)
            if personalized_questions:
                return personalized_questions

        # If no weak areas or personalized questions failed, get general questions
        cache_key = self._get_cache_key(f"quiz_{topic}_{level}", topic)
        
        # Try to get from cache first
        cached_questions = cache.get(cache_key)
        if cached_questions:
            logger.debug(f"Retrieved quiz questions from cache for topic: {topic}, level: {level}")
            return cached_questions

        try:
            # Prepare the prompt for general questions
            prompt = f"""Generate 10 multiple-choice questions for a {topic} quiz at level {level}.
            Each question should have:
            1. A clear question stem
            2. 4 answer choices (A, B, C, D)
            3. One correct answer
            4. Difficulty appropriate for level {level}
            
            Format each question as JSON with these fields:
            - question_text: The question stem
            - choices: List of 4 answer choices
            - correct_answer: The correct choice (A, B, C, or D)
            - explanation: Brief explanation of the correct answer
            - points: Difficulty points (1-5)
            
            Return only the JSON array of questions."""

            # Generate response using Ollama with Mistral
            response = self._generate_with_ollama(prompt)
            
            # Parse the response
            try:
                questions = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    questions = json.loads(json_match.group())
                else:
                    logger.error("Failed to parse Ollama response as JSON")
                    return []

            # Cache the results
            cache.set(cache_key, questions, self.cache_timeout)
            logger.debug(f"Cached quiz questions for topic: {topic}, level: {level}")
            
            return questions

        except Exception as e:
            logger.error(f"Error generating quiz questions: {str(e)}")
            return []

    def _filter_inappropriate_content(self, results: List[Dict]) -> List[Dict]:
        """
        Filter out inappropriate content based on keywords.
        """
        filtered_results = []
        for result in results:
            title = result.get('title', '').lower()
            description = result.get('description', '').lower()
            
            # Check if content contains inappropriate keywords
            is_appropriate = not any(keyword in title or keyword in description 
                                  for keyword in self.inappropriate_keywords)
            
            if is_appropriate:
                filtered_results.append(result)
        
        return filtered_results

    def _score_and_sort_results(self, results: List[Dict]) -> List[Dict]:
        """
        Score and sort results based on various quality metrics.
        """
        scored_results = []
        for result in results:
            score = 0.0
            
            # Base score from source reliability
            source = result.get('source', '')
            if source in self.sources:
                score += self.sources[source]['reliability_score']
            
            # Quality indicators
            if result.get('description'):
                score += 0.1
            if result.get('author'):
                score += 0.1
            if result.get('date'):
                score += 0.1
            if result.get('citations', 0) > 0:
                score += min(result['citations'] * 0.01, 0.3)
            
            # Penalize short content
            if len(result.get('description', '')) < 50:
                score -= 0.2
            
            result['quality_score'] = min(max(score, 0), 1)
            scored_results.append(result)
        
        # Sort by quality score
        return sorted(scored_results, key=lambda x: x['quality_score'], reverse=True)

    def create_study_links(self, question, subtopic) -> List[StudyLink]:
        """Create study links from educational content using KNN"""
        try:
            query = f"{subtopic.name} {subtopic.general_topic.name} {question.question_text}"
            content_list = self.search_educational_content(query)
            
            study_links = []
            for content in content_list[:3]:  # Limit to top 3 results
                study_link = StudyLink.objects.create(
                    title=f"{content['source']}: {content['title']}",
                    description=content['description'],
                    url=content['url'],
                    subtopic=subtopic
                )
                study_links.append(study_link)
            
            return study_links
        except Exception as e:
            logger.error(f"Error in create_study_links: {str(e)}")
            return []

    def generate_study_materials(self, subject: str, material_type: str, difficulty: str, weak_areas: Dict = None) -> List[Dict]:
        """
        Generate study materials using Khan Academy and direct searches to Google/YouTube.
        """
        try:
            study_materials = []
            
            # Always include Khan Academy materials
            khan_query = f"{subject} {difficulty}"
            khan_url = f"https://www.khanacademy.org/search?page_search_query={quote_plus(khan_query)}"
            
            study_materials.append({
                'title': f"Khan Academy - {subject}",
                'description': f"Comprehensive {difficulty} level resources for {subject}",
                'url': khan_url,
                'material_type': 'text',
                'difficulty': difficulty,
                'source': 'Khan Academy'
            })
            
            # For video materials, search YouTube directly
            if material_type == 'video':
                youtube_query = f"{subject} {difficulty} tutorial"
                youtube_url = f"https://www.youtube.com/results?search_query={quote_plus(youtube_query)}"
                
                study_materials.append({
                    'title': f"YouTube Tutorials - {subject}",
                    'description': f"Video tutorials for {subject} at {difficulty} level",
                    'url': youtube_url,
                    'material_type': 'video',
                    'difficulty': difficulty,
                    'source': 'YouTube'
                })
            
            # For text materials, search Google Scholar
            if material_type == 'text':
                scholar_query = f"{subject} {difficulty} study guide"
                scholar_url = f"https://scholar.google.com/scholar?q={quote_plus(scholar_query)}"
                
                study_materials.append({
                    'title': f"Study Guides - {subject}",
                    'description': f"Academic resources and study guides for {subject}",
                    'url': scholar_url,
                    'material_type': 'text',
                    'difficulty': difficulty,
                    'source': 'Google Scholar'
                })
            
            # For practice problems
            if material_type == 'practice':
                practice_query = f"{subject} {difficulty} practice problems"
                practice_url = f"https://www.google.com/search?q={quote_plus(practice_query)}"
                
                study_materials.append({
                    'title': f"Practice Problems - {subject}",
                    'description': f"Practice problems and exercises for {subject}",
                    'url': practice_url,
                    'material_type': 'practice',
                    'difficulty': difficulty,
                    'source': 'Google'
                })
            
            # For interactive materials
            if material_type == 'interactive':
                interactive_query = f"{subject} {difficulty} interactive learning"
                interactive_url = f"https://www.google.com/search?q={quote_plus(interactive_query)}"
                
                study_materials.append({
                    'title': f"Interactive Learning - {subject}",
                    'description': f"Interactive learning resources for {subject}",
                    'url': interactive_url,
                    'material_type': 'interactive',
                    'difficulty': difficulty,
                    'source': 'Google'
                })
            
            # For additional quizzes
            if material_type == 'quiz':
                quiz_query = f"{subject} {difficulty} quiz practice"
                quiz_url = f"https://www.google.com/search?q={quote_plus(quiz_query)}"
                
                study_materials.append({
                    'title': f"Practice Quizzes - {subject}",
                    'description': f"Additional quiz questions for {subject}",
                    'url': quiz_url,
                    'material_type': 'quiz',
                    'difficulty': difficulty,
                    'source': 'Google'
                })
            
            # Add weak areas specific materials if available
            if weak_areas:
                for material in study_materials:
                    if weak_areas.get('general_topics'):
                        topics = GeneralTopic.objects.filter(id__in=weak_areas['general_topics'])
                        material['description'] += f" (Focus areas: {', '.join(topic.name for topic in topics)})"
                    
                    if weak_areas.get('subtopics'):
                        subtopics = Subtopic.objects.filter(id__in=weak_areas['subtopics'])
                        material['description'] += f" (Specific topics: {', '.join(subtopic.name for subtopic in subtopics)})"
            
            return study_materials
            
        except Exception as e:
            logger.error(f"Error generating study materials: {str(e)}")
            return []

    def get_next_quiz_level(self, student_id: int, subject: str) -> int:
        """
        Determine the next quiz level for a student based on their performance.
        Returns 1 if no previous attempts, or next level if previous level was passed.
        """
        try:
            # Get the latest performance record for the student
            latest_performance = StudentPerformance.objects.filter(
                student_id=student_id,
                subject=subject
            ).order_by('-created_at').first()

            if not latest_performance:
                return 1  # Start with level 1 if no previous attempts

            # If the student passed the last level, move to next level
            if latest_performance.passed:
                return latest_performance.level + 1
            else:
                # Stay at the same level if not passed
                return latest_performance.level

        except Exception as e:
            logger.error(f"Error determining next quiz level: {str(e)}")
            return 1

    def get_weak_areas(self, student_id: int, subject: str) -> Dict:
        """
        Analyze student's performance to identify weak areas.
        Returns a dictionary with weak general topics and subtopics.
        """
        try:
            # Get all performance records for the student in this subject
            performances = StudentPerformance.objects.filter(
                student_id=student_id,
                subject=subject
            )

            # Initialize counters for topics and subtopics
            topic_scores = {}
            subtopic_scores = {}

            # Calculate average scores for each topic and subtopic
            for perf in performances:
                for topic in perf.weak_areas.get('general_topics', []):
                    topic_scores[topic] = topic_scores.get(topic, 0) + 1
                for subtopic in perf.weak_areas.get('subtopics', []):
                    subtopic_scores[subtopic] = subtopic_scores.get(subtopic, 0) + 1

            # Identify weak areas (topics/subtopics with highest error counts)
            weak_general_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            weak_subtopics = sorted(subtopic_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                'general_topics': [topic[0] for topic in weak_general_topics],
                'subtopics': [subtopic[0] for subtopic in weak_subtopics]
            }

        except Exception as e:
            logger.error(f"Error analyzing weak areas: {str(e)}")
            return {'general_topics': [], 'subtopics': []}

    def generate_level_quiz(self, student_id: int, subject: str, level: int) -> Dict:
        """
        Generate a quiz for a specific level, focusing on weak areas for levels > 1.
        """
        try:
            # Get weak areas for levels > 1
            weak_areas = self.get_weak_areas(student_id, subject) if level > 1 else None

            if level == 1:
                # Level 1: General questions from database
                questions = self.get_quiz_questions(subject, level)
            else:
                # Higher levels: Focus on weak areas
                questions = self.generate_personalized_questions(
                    weak_areas=weak_areas['subtopics'],
                    subject=subject,
                    level=level
                )

            # Generate study materials based on weak areas
            study_materials = self.generate_study_materials(
                subject=subject,
                material_type='text',  # Default to text, can be modified based on student preference
                difficulty=f'level_{level}',
                weak_areas=weak_areas
            )

            return {
                'level': level,
                'questions': questions,
                'study_materials': study_materials,
                'weak_areas': weak_areas
            }

        except Exception as e:
            logger.error(f"Error generating level quiz: {str(e)}")
            return {
                'level': level,
                'questions': [],
                'study_materials': [],
                'weak_areas': {}
            }

    def evaluate_quiz_performance(self, student_id: int, subject: str, level: int, 
                                answers: List[Dict]) -> Dict:
        """
        Evaluate student's quiz performance and determine if they passed the level.
        If failed, provide study materials and prepare for next level focused on weak areas.
        """
        try:
            # Calculate score
            total_questions = len(answers)
            correct_answers = sum(1 for ans in answers if ans.get('is_correct', False))
            score = (correct_answers / total_questions) * 100

            # Determine if passed (e.g., 70% or higher)
            passed = score >= 70

            # Identify weak areas from incorrect answers
            weak_areas = {
                'general_topics': [],
                'subtopics': []
            }

            for ans in answers:
                if not ans.get('is_correct', False):
                    question = Question.objects.get(id=ans.get('question_id'))
                    if question.subtopic.general_topic.id not in weak_areas['general_topics']:
                        weak_areas['general_topics'].append(question.subtopic.general_topic.id)
                    if question.subtopic.id not in weak_areas['subtopics']:
                        weak_areas['subtopics'].append(question.subtopic.id)

            # Save performance record
            StudentPerformance.objects.create(
                student_id=student_id,
                subject=subject,
                level=level,
                score=score,
                passed=passed,
                weak_areas=weak_areas
            )

            # Prepare response with study materials if failed
            response = {
                'score': score,
                'passed': passed,
                'weak_areas': weak_areas,
                'next_level': level + 1 if passed else level
            }

            if not passed:
                # Generate focused study materials for weak areas
                study_materials = self.generate_focused_study_materials(
                    subject=subject,
                    weak_areas=weak_areas,
                    current_level=level
                )
                response['study_materials'] = study_materials

                # Prepare next level quiz focused on weak areas
                next_quiz = self.generate_level_quiz(
                    student_id=student_id,
                    subject=subject,
                    level=level + 1
                )
                response['next_quiz_preview'] = {
                    'level': next_quiz['level'],
                    'focus_areas': next_quiz['weak_areas']
                }

            return response

        except Exception as e:
            logger.error(f"Error evaluating quiz performance: {str(e)}")
            return {
                'score': 0,
                'passed': False,
                'weak_areas': {},
                'next_level': level
            }

    def generate_focused_study_materials(self, subject: str, weak_areas: Dict, 
                                       current_level: int) -> List[Dict]:
        """
        Generate focused study materials based on weak areas and current level.
        """
        try:
            study_materials = []
            
            # Generate materials for each weak area
            for topic_id in weak_areas['general_topics']:
                topic = GeneralTopic.objects.get(id=topic_id)
                
                # Text-based materials
                text_materials = self.generate_study_materials(
                    subject=subject,
                    material_type='text',
                    difficulty=f'level_{current_level}',
                    weak_areas={'general_topics': [topic_id]}
                )
                study_materials.extend(text_materials)
                
                # Video materials
                video_materials = self.generate_study_materials(
                    subject=subject,
                    material_type='video',
                    difficulty=f'level_{current_level}',
                    weak_areas={'general_topics': [topic_id]}
                )
                study_materials.extend(video_materials)

            for subtopic_id in weak_areas['subtopics']:
                subtopic = Subtopic.objects.get(id=subtopic_id)
                
                # Practice problems
                practice_materials = self.generate_study_materials(
                    subject=subject,
                    material_type='practice',
                    difficulty=f'level_{current_level}',
                    weak_areas={'subtopics': [subtopic_id]}
                )
                study_materials.extend(practice_materials)
                
                # Interactive materials
                interactive_materials = self.generate_study_materials(
                    subject=subject,
                    material_type='interactive',
                    difficulty=f'level_{current_level}',
                    weak_areas={'subtopics': [subtopic_id]}
                )
                study_materials.extend(interactive_materials)

            # Add additional quiz practice
            quiz_materials = self.generate_study_materials(
                subject=subject,
                material_type='quiz',
                difficulty=f'level_{current_level}',
                weak_areas=weak_areas
            )
            study_materials.extend(quiz_materials)

            # Remove duplicates and sort by material type
            unique_materials = []
            seen_urls = set()
            for material in study_materials:
                if material['url'] not in seen_urls:
                    seen_urls.add(material['url'])
                    unique_materials.append(material)

            return sorted(unique_materials, key=lambda x: x['material_type'])

        except Exception as e:
            logger.error(f"Error generating focused study materials: {str(e)}")
            return []

    def _generate_default_materials(self, subject: str, material_type: str) -> List[Dict]:
        """Generate default study materials when models fail"""
        default_materials = []
        
        # Generate default materials based on material type
        if material_type == 'video':
            default_materials.append({
                'title': f"Video Tutorial: {subject}",
                'description': f"Learn {subject} through video tutorials",
                'url': f"https://www.youtube.com/results?search_query={subject}+tutorial",
                'material_type': 'video',
                'source': 'YouTube',
                'relevance_score': 0.8
            })
        elif material_type == 'text':
            default_materials.append({
                'title': f"Text Guide: {subject}",
                'description': f"Read about {subject}",
                'url': f"https://www.khanacademy.org/search?page_search_query={subject}",
                'material_type': 'text',
                'source': 'Khan Academy',
                'relevance_score': 0.8
            })
        elif material_type == 'interactive':
            default_materials.append({
                'title': f"Interactive Lesson: {subject}",
                'description': f"Practice {subject} interactively",
                'url': f"https://www.khanacademy.org/search?page_search_query={subject}+practice",
                'material_type': 'interactive',
                'source': 'Khan Academy',
                'relevance_score': 0.8
            })
        
        return default_materials 