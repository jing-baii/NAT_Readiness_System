import unittest
from unittest.mock import patch, MagicMock
from core.models import Subtopic, GeneralTopic, Topic
from core.utils import generate_questions_with_ollama  # adjust the import path

class TestGenerateQuestionsWithOllama(unittest.TestCase):

    @patch('core.utils.requests.post')  # Mocking requests.post
    def test_generate_questions_with_ollama(self, mock_post):
        # Arrange
        subtopic_name = "Subtopic Test"
        level_number = 1
        subtopic = MagicMock(spec=Subtopic)
        subtopic.name = subtopic_name
        subtopic.general_topic.name = "General Topic Test"

        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': '''Question: What is the capital of France?
            a) Paris
            b) London
            c) Berlin
            d) Madrid
            Correct Answer: a
            Explanation: Paris is the capital of France.
            
            Question: What is the largest ocean?
            a) Atlantic Ocean
            b) Indian Ocean
            c) Arctic Ocean
            d) Pacific Ocean
            Correct Answer: d
            Explanation: The Pacific Ocean is the largest ocean in the world.
            '''
        }
        mock_post.return_value = mock_response

        # Act
        questions = generate_questions_with_ollama(subtopic, level_number)

        # Assert
        self.assertEqual(len(questions), 2)  # Should return two questions
        self.assertEqual(questions[0]['text'], "What is the capital of France?")
        self.assertEqual(questions[0]['choices'], ["Paris", "London", "Berlin", "Madrid"])
        self.assertEqual(questions[0]['correct_answer'], "a")
        self.assertEqual(questions[0]['explanation'], "Paris is the capital of France.")
        
        self.assertEqual(questions[1]['text'], "What is the largest ocean?")
        self.assertEqual(questions[1]['choices'], ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"])
        self.assertEqual(questions[1]['correct_answer'], "d")
        self.assertEqual(questions[1]['explanation'], "The Pacific Ocean is the largest ocean in the world.")

    @patch('core.utils.requests.post')  # Mocking requests.post
    def test_generate_questions_with_ollama_api_error(self, mock_post):
        # Arrange
        subtopic_name = "Subtopic Test"
        level_number = 1
        subtopic = MagicMock(spec=Subtopic)
        subtopic.name = subtopic_name
        subtopic.general_topic.name = "General Topic Test"

        # Mock the API response to simulate an error
        mock_post.side_effect = requests.exceptions.Timeout

        # Act
        questions = generate_questions_with_ollama(subtopic, level_number)

        # Assert
        self.assertEqual(questions, [])

    @patch('core.utils.requests.post')  # Mocking requests.post
    def test_generate_questions_with_ollama_invalid_response(self, mock_post):
        # Arrange
        subtopic_name = "Subtopic Test"
        level_number = 1
        subtopic = MagicMock(spec=Subtopic)
        subtopic.name = subtopic_name
        subtopic.general_topic.name = "General Topic Test"

        # Mock the API response with invalid data
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'Invalid Response'}
        mock_post.return_value = mock_response

        # Act
        questions = generate_questions_with_ollama(subtopic, level_number)

        # Assert
        self.assertEqual(questions, [])

if __name__ == '__main__':
    unittest.main()
