import unittest
from django.test import TestCase
from django.core.cache import cache
from ..services.educational_content import EducationalContentService
from ..models import Question, Subtopic, GeneralTopic, Topic, StudyLink

class TestEducationalContentService(TestCase):
    def setUp(self):
        self.service = EducationalContentService()
        self.topic = Topic.objects.create(
            name="Mathematics",
            description="Mathematics subject"
        )
        self.general_topic = GeneralTopic.objects.create(
            name="Algebra",
            subject=self.topic
        )
        self.subtopic = Subtopic.objects.create(
            name="Quadratic Equations",
            general_topic=self.general_topic
        )
        self.question = Question.objects.create(
            question_text="What is the quadratic formula?",
            subtopic=self.subtopic
        )
        cache.clear()

    def test_search_educational_content(self):
        # Test basic search
        results = self.service.search_educational_content("quadratic formula")
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        
        # Verify result structure
        for result in results:
            self.assertIn('title', result)
            self.assertIn('description', result)
            self.assertIn('url', result)
            self.assertIn('source', result)
            self.assertIn('type', result)
            self.assertIn('reliability_score', result)
            self.assertIn('quality_score', result)
            
            # Verify URL format
            self.assertTrue(result['url'].startswith(('http://', 'https://')))
            
            # Verify scores are within range
            self.assertTrue(0 <= result['reliability_score'] <= 1)
            self.assertTrue(0 <= result['quality_score'] <= 1)

    def test_search_with_material_type(self):
        # Test video search
        results = self.service.search_educational_content("quadratic formula", material_type="video")
        self.assertIsInstance(results, list)
        
        # Verify video results
        video_results = [r for r in results if r['type'] == 'video']
        self.assertTrue(len(video_results) > 0)
        
        for video in video_results:
            self.assertEqual(video['type'], 'video')
            self.assertIn('youtube.com', video['url'])

    def test_content_filtering(self):
        # Test inappropriate content filtering
        results = self.service.search_educational_content("test inappropriate content")
        self.assertIsInstance(results, list)
        
        # Verify no inappropriate content
        for result in results:
            title = result['title'].lower()
            description = result['description'].lower()
            self.assertFalse(any(keyword in title or keyword in description 
                               for keyword in self.service.inappropriate_keywords))

    def test_caching(self):
        # Test caching functionality
        query = "test caching"
        results1 = self.service.search_educational_content(query)
        results2 = self.service.search_educational_content(query)
        
        # Verify results are identical (from cache)
        self.assertEqual(results1, results2)
        
        # Verify cache key exists
        cache_key = f'educational_content_{query}_None'
        self.assertIsNotNone(cache.get(cache_key))

    def test_rate_limiting(self):
        # Test rate limiting
        for _ in range(5):  # Make multiple requests
            results = self.service.search_educational_content("test rate limit")
            self.assertIsInstance(results, list)

    def test_quality_scoring(self):
        # Test quality scoring
        results = self.service.search_educational_content("test quality scoring")
        self.assertIsInstance(results, list)
        
        # Verify results are sorted by quality score
        scores = [r['quality_score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Verify quality score components
        for result in results:
            score = 0.0
            
            # Source reliability
            if result['source'] in self.service.sources:
                score += self.service.sources[result['source']]['reliability_score']
            
            # Content completeness
            if result.get('description'):
                score += 0.1
            if result.get('author'):
                score += 0.1
            if result.get('date'):
                score += 0.1
            if result.get('citations', 0) > 0:
                score += min(result['citations'] * 0.01, 0.3)
            
            # Content length penalty
            if len(result.get('description', '')) < 50:
                score -= 0.2
            
            # Verify calculated score matches stored score
            self.assertAlmostEqual(score, result['quality_score'], places=2)

    def test_create_study_links(self):
        # Test study link creation
        study_links = self.service.create_study_links(self.question, self.subtopic)
        self.assertIsInstance(study_links, list)
        
        # Verify study link structure
        for link in study_links:
            self.assertIsInstance(link, StudyLink)
            self.assertEqual(link.subtopic, self.subtopic)
            self.assertTrue(link.url.startswith(('http://', 'https://')))
            self.assertTrue(len(link.title) > 0)
            self.assertTrue(len(link.description) > 0)

    def test_ollama_prompt_format(self):
        # Patch the _generate_with_ollama method to capture the prompt
        prompts = []
        def fake_generate_with_ollama(prompt, max_length=1000):
            prompts.append(prompt)
            # Return a minimal valid JSON array as a string
            return '[{"question_text": "Sample?", "choices": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "Sample explanation.", "points": 1}]'
        self.service._generate_with_ollama = fake_generate_with_ollama

        weak_areas = ["Quadratic Equations", "Factoring"]
        subject = "Mathematics"
        level = 2
        num_questions = 3
        self.service.generate_personalized_questions(weak_areas, subject, level, num_questions=num_questions)

        self.assertTrue(prompts)
        prompt = prompts[0]
        self.assertIn(f"Generate {num_questions} multiple-choice questions", prompt)
        self.assertIn(subject, prompt)
        self.assertIn(f"level {level}", prompt)
        for area in weak_areas:
            self.assertIn(area, prompt)
        self.assertIn("Format each question as JSON", prompt)
        self.assertIn("Return only the JSON array of questions.", prompt)

    def test_ollama_offline_generation(self):
        """Integration test: Calls Ollama locally to generate questions for weak areas and prints prompt/response."""
        weak_areas = ["Quadratic Equations", "Factoring"]
        subject = "Mathematics"
        level = 2
        num_questions = 2

        # Patch the _generate_with_ollama method to capture prompt and response
        original_generate_with_ollama = self.service._generate_with_ollama
        captured = {}
        def capture_generate_with_ollama(prompt, max_length=1000):
            captured['prompt'] = prompt
            response = original_generate_with_ollama(prompt, max_length)
            captured['response'] = response
            return response
        self.service._generate_with_ollama = capture_generate_with_ollama

        questions = self.service.generate_personalized_questions(
            weak_areas, subject, level, num_questions=num_questions
        )

        # Print the prompt and raw response for inspection
        print("\n--- PROMPT SENT TO OLLAMA ---\n", captured.get('prompt', 'No prompt captured'))
        print("\n--- RAW RESPONSE FROM OLLAMA ---\n", captured.get('response', 'No response captured'))

        self.assertIsInstance(questions, list)
        self.assertGreater(len(questions), 0)
        for q in questions:
            self.assertIn('question_text', q)
            self.assertIn('choices', q)
            self.assertIn('correct_answer', q)
            self.assertIn('explanation', q)
            self.assertIn('points', q)

if __name__ == '__main__':
    unittest.main() 