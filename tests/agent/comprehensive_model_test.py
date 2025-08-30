#!/usr/bin/env python3
"""
Comprehensive Model Testing Script
Runs test prompts from easy/medium/hard files on multiple models and architectures with rate limiting and logging.
Results are organized by architecture and model in separate directories for easy comparison.
"""

import sys
import os
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Import the Gemini API rotation manager
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gemini_api_rotation import GeminiApiRotationManager

# Force unbuffered output for nohup compatibility
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Starting script initialization...", flush=True)
print(f"Python path: {sys.path[0]}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)

try:
    from biomni.agent import A1
    from biomni.agent.architecture.b1 import B1
    from biomni.agent.architecture.c1 import C1
    from biomni.agent.architecture.d1 import D1
    from biomni.agent.architecture.e1 import E1
    from biomni.agent.architecture.f1 import F1
    print("Successfully imported all agent architectures: A1, B1, C1, D1, E1, F1", flush=True)
except ImportError as e:
    print(f"FATAL ERROR: Could not import agent architectures: {e}", flush=True)
    print("Make sure you're in the correct directory and biomni is installed", flush=True)
    sys.exit(1)


class ModelTester:
    def __init__(self):
        print("Initializing ModelTester...", flush=True)
        
        # Initialize Gemini API rotation manager
        try:
            self.gemini_rotation_manager = GeminiApiRotationManager(
                cooldown_minutes=60,  # 1 hour cooldown for rate-limited keys
                max_retries_per_key=3,
                retry_delay_seconds=5,
                log_level="INFO"
            )
            print(f"Initialized Gemini API rotation with {len(self.gemini_rotation_manager.api_keys)} keys", flush=True)
        except Exception as e:
            print(f"Warning: Could not initialize Gemini API rotation: {e}", flush=True)
            self.gemini_rotation_manager = None
        
        # Check environment variables early
        base_url = os.environ.get("CUSTOM_MODEL_BASE_URL")
        api_key = os.environ.get("CUSTOM_MODEL_API_KEY")
        claude_api_key = os.environ.get("CLAUDE_API_KEY")  # Add Claude key

        # Set default model name for database tools if not set
        if not os.environ.get("CUSTOM_MODEL_NAME"):
            os.environ["CUSTOM_MODEL_NAME"] = "qwen3-235b-a22b"  # Use a default model
            print("Set default CUSTOM_MODEL_NAME for database tools", flush=True)
        
        if not base_url or not api_key:
            print("WARNING: CUSTOM_MODEL_BASE_URL or CUSTOM_MODEL_API_KEY not set!", flush=True)
            print("Custom models may fail without these environment variables", flush=True)
        else:
            print(f"Environment check passed. Base URL: {base_url[:50]}...", flush=True)

        if not claude_api_key:
            print("WARNING: CLAUDE_API_KEY not set for database tools!", flush=True)
        else:
            print("Claude API key configured for database tools", flush=True)
            # Set it as environment variable so database tools can access it
            os.environ["CLAUDE_API_KEY"] = claude_api_key
        
        # Define agent architectures
        self.architectures = [
            {
                'name': 'A1',
                'class': A1,
                'description': 'Original A1 Architecture'
            },
                # {
                #     'name': 'B1', 
                #     'class': B1,
                #     'description': 'Hierarchical Expert Model'
                # },
                # {
                #     'name': 'C1',
                #     'class': C1, 
                #     'description': 'Cognitive Corrector Model'
                # },
                # {
                #     'name': 'D1',
                #     'class': D1,
                #     'description': 'Exploratory Sandbox Model'
                # },
            # {
            #     'name': 'E1',
            #     'class': E1,
            #     'description': 'Tool-Augmented Graph (TAG) Model'
            # },
            # {
            #     'name': 'F1',
            #     'class': F1,
            #     'description': 'Advanced Architecture Model'
            # }
        ]
        
        self.models = [
            {
                'name': 'gemini-2.5-pro',
                'type': 'gemini',
                'rate_limit': 5,  # 5 calls per minute
                'config': {
                    'llm': 'gemini-2.5-pro',
                    'api_key': self._get_gemini_api_key(),  # Use rotation manager
                    'timeout_seconds': 1000
                }
            },
            {
                'name': 'qwen3-235b-a22b',
                'type': 'custom',
                'rate_limit': None,
                'config': {
                    'llm': 'custom',
                    'base_url': os.environ.get("CUSTOM_MODEL_BASE_URL"),
                    'api_key': os.environ.get("CUSTOM_MODEL_API_KEY"),
                    'custom_model_name': 'qwen3-235b-a22b',
                    'timeout_seconds': 1000
                }
            },
            # {
            #     'name': 'codestral-22b',
            #     'type': 'custom',
            #     'rate_limit': None,
            #     'config': {
            #         'llm': 'custom',
            #         'base_url': os.environ.get("CUSTOM_MODEL_BASE_URL"),
            #         'api_key': os.environ.get("CUSTOM_MODEL_API_KEY"),
            #         'custom_model_name': 'codestral-22b',
            #         'timeout_seconds': 1000
            #     }
            # },
            # {
            #     'name': 'qwen2.5-coder-32b-instruct',
            #     'type': 'custom',
            #     'rate_limit': None,
            #     'config': {
            #         'llm': 'custom',
            #         'base_url': os.environ.get("CUSTOM_MODEL_BASE_URL"),
            #         'api_key': os.environ.get("CUSTOM_MODEL_API_KEY"),
            #         'custom_model_name': 'qwen2.5-coder-32b-instruct',
            #         'timeout_seconds': 1000
            #     }
            # }
        ]
        
        self.prompts = self.load_prompts()
        
        # Create organized directory structure
        self.base_log_dir = Path("test_results")
        self.base_log_dir.mkdir(exist_ok=True)
        self.setup_directory_structure()
        
        # Track API call timing for rate limiting
        self.last_api_calls = {}
        
        # Judge configuration - using Gemini Pro 2.5 as the judge
        self.judge_config = {
            'llm': 'gemini-2.5-pro',
            'api_key': self._get_gemini_api_key(),  # Use rotation manager
            'timeout_seconds': 1000
        }
    
    def _get_gemini_api_key(self) -> str:
        """Get current Gemini API key from rotation manager"""
        if self.gemini_rotation_manager:
            try:
                return self.gemini_rotation_manager.get_current_api_key()
            except Exception as e:
                print(f"Warning: Could not get API key from rotation manager: {e}", flush=True)
                # Fallback to environment variable
                return os.environ.get("GEMINI_API_KEY")
        else:
            return os.environ.get("GEMINI_API_KEY")
    
    def _handle_gemini_api_error(self, error: Exception, response_code: Optional[int] = None) -> bool:
        """Handle Gemini API errors and attempt rotation if needed"""
        if self.gemini_rotation_manager:
            try:
                should_retry = self.gemini_rotation_manager.handle_api_error(error, response_code)
                if should_retry:
                    print(f"API key rotated, will retry with new key", flush=True)
                    self.gemini_rotation_manager.print_status()
                return should_retry
            except Exception as e:
                print(f"Warning: Error in API rotation handling: {e}", flush=True)
        return False
    
    def _record_successful_gemini_request(self):
        """Record a successful Gemini API request"""
        if self.gemini_rotation_manager:
            self.gemini_rotation_manager.record_successful_request()
    
    def setup_directory_structure(self):
        """Create organized directory structure for results"""
        print("Setting up directory structure...", flush=True)
        
        # Create main directories
        self.directories = {}
        
        for architecture in self.architectures:
            arch_name = architecture['name']
            arch_dir = self.base_log_dir / arch_name
            arch_dir.mkdir(exist_ok=True)
            
            self.directories[arch_name] = {}
            
            for model in self.models:
                model_name = model['name']
                model_dir = arch_dir / model_name
                model_dir.mkdir(exist_ok=True)
                
                # Create subdirectories for different types of outputs
                (model_dir / "individual_tests").mkdir(exist_ok=True)
                (model_dir / "summaries").mkdir(exist_ok=True)
                (model_dir / "logs").mkdir(exist_ok=True)
                (model_dir / "quality_evaluations").mkdir(exist_ok=True)
                
                self.directories[arch_name][model_name] = model_dir
                
                print(f"Created directory: {model_dir}", flush=True)
        
        # Create comparison directories
        comparison_dir = self.base_log_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        (comparison_dir / "by_architecture").mkdir(exist_ok=True)
        (comparison_dir / "by_model").mkdir(exist_ok=True)
        (comparison_dir / "by_difficulty").mkdir(exist_ok=True)
        (comparison_dir / "overall").mkdir(exist_ok=True)
        (comparison_dir / "quality_analysis").mkdir(exist_ok=True)
        
        self.comparison_dir = comparison_dir
        print(f"Created comparison directory: {comparison_dir}", flush=True)

    def load_prompts(self) -> Dict[str, List[str]]:
        """Load prompts from the three difficulty files"""
        prompts = {}
        
        # Files are in project root (two directories up from this script)
        root_dir = Path(__file__).parent.parent.parent
        files = {
            'easy': root_dir / 'test_prompts_easy.txt',
            'medium': root_dir / 'test_prompts_medium.txt', 
            'hard': root_dir / 'test_prompts_hard.txt'
        }
        
        for difficulty, filename in files.items():
            prompts[difficulty] = []
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line:  # Skip empty lines
                            prompts[difficulty].append(line)
                print(f"Loaded {len(prompts[difficulty])} {difficulty} prompts", flush=True)
            except FileNotFoundError:
                print(f"Warning: {filename} not found", flush=True)
                prompts[difficulty] = []
        
        return prompts
    
    def create_agent(self, architecture: Dict[str, Any], model_config: Dict[str, Any]):
        """Create an agent with the specified architecture and model configuration"""
        config = {
            'path': '/mnt/exchange-saia/protein/haldhah/biomni_datalake',
            'use_tool_retriever': True,
            **model_config
        }
        
        # Update Gemini API key if this is a Gemini model
        if config.get('llm') == 'gemini-2.5-pro':
            config['api_key'] = self._get_gemini_api_key()
        
        # Handle F1 architecture with its architecture parameter
        if architecture['name'] == 'F1':
            config['architecture'] = 'baseline'  # Default F1 architecture
        
        agent_class = architecture['class']
        return agent_class(**config)
    
    def wait_for_rate_limit(self, model_name: str, rate_limit: int):
        """Wait if necessary to respect rate limits"""
        if rate_limit is None:
            return
            
        current_time = time.time()
        if model_name in self.last_api_calls:
            time_since_last = current_time - self.last_api_calls[model_name]
            min_interval = 60.0 / rate_limit  # seconds between calls
            
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last + 10  # +1 second buffer
                print(f"Rate limiting: waiting {wait_time:.1f}s for {model_name}", flush=True)
                time.sleep(wait_time)
        
        self.last_api_calls[model_name] = time.time()
    
    def create_judge_agent(self):
        """Create a judge agent using Gemini Pro 2.5"""
        try:
            from biomni.agent import A1
            config = {
                'path': '/mnt/exchange-saia/protein/haldhah/biomni_datalake',
                'use_tool_retriever': False,  # Judge doesn't need tools
                **self.judge_config
            }
            # Update API key for judge agent
            config['api_key'] = self._get_gemini_api_key()
            return A1(**config)
        except Exception as e:
            print(f"Failed to create judge agent: {e}", flush=True)
            return None
    
    def create_quality_assessment_prompt(self, prompt: str, baseline_response: str, test_response: str, difficulty: str) -> str:
        """Create a prompt for quality assessment and hallucination detection"""
        return f"""You are an expert evaluator assessing the quality of AI responses to scientific questions. 

TASK: Compare two responses to the same scientific question and provide a detailed quality assessment.

QUESTION ({difficulty.upper()} difficulty):
{prompt}

BASELINE RESPONSE (A1 + Gemini Pro 2.5):
{baseline_response}

TEST RESPONSE (Architecture under evaluation):
{test_response}

Please evaluate the TEST RESPONSE compared to the BASELINE RESPONSE on the following criteria:

1. **FACTUAL ACCURACY** (0-10): Are the scientific facts, data, and claims accurate?
2. **COMPLETENESS** (0-10): Does it address all aspects of the question adequately?
3. **SCIENTIFIC RIGOR** (0-10): Is the methodology and reasoning scientifically sound?
4. **CLARITY** (0-10): Is the response clear, well-structured, and understandable?
5. **HALLUCINATION DETECTION** (0-10): Rate the presence of hallucinations (0=many hallucinations, 10=no hallucinations)

For each criterion, provide:
- A numerical score (0-10)
- A brief explanation (1-2 sentences)

Then provide:
- **OVERALL QUALITY SCORE** (0-10): Average of all criteria
- **HALLUCINATION RISK** (LOW/MEDIUM/HIGH): Based on factual accuracy and scientific rigor
- **COMPARISON VERDICT** (BETTER/EQUAL/WORSE): How does the test response compare to baseline?
- **KEY DIFFERENCES**: List 2-3 main differences between the responses
- **IMPROVEMENT SUGGESTIONS**: 1-2 specific suggestions for the test response

Format your response as structured JSON:
```json
{{
    "factual_accuracy": {{"score": X, "explanation": "..."}},
    "completeness": {{"score": X, "explanation": "..."}},
    "scientific_rigor": {{"score": X, "explanation": "..."}},
    "clarity": {{"score": X, "explanation": "..."}},
    "hallucination_detection": {{"score": X, "explanation": "..."}},
    "overall_quality_score": X.X,
    "hallucination_risk": "LOW/MEDIUM/HIGH",
    "comparison_verdict": "BETTER/EQUAL/WORSE",
    "key_differences": ["...", "...", "..."],
    "improvement_suggestions": ["...", "..."]
}}
```"""

    def evaluate_response_quality(self, prompt: str, baseline_response: str, test_response: str, 
                                difficulty: str, test_arch: str, test_model: str) -> Optional[Dict]:
        """Evaluate response quality using LLM judge"""
        judge_agent = self.create_judge_agent()
        if not judge_agent:
            return None
        
        try:
            # Rate limit for judge calls
            self.wait_for_rate_limit('judge', 5)  # 5 calls per minute for judge
            
            assessment_prompt = self.create_quality_assessment_prompt(
                prompt, baseline_response, test_response, difficulty
            )
            
            print(f"Evaluating {test_arch}+{test_model} response quality...", flush=True)
            
            start_time = time.time()
            judge_response = judge_agent.go(assessment_prompt)
            evaluation_time = time.time() - start_time
            
            # Extract JSON from response
            judge_text = str(judge_response)
            try:
                # Find JSON block in response
                json_start = judge_text.find('{')
                json_end = judge_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = judge_text[json_start:json_end]
                    evaluation_result = json.loads(json_str)
                    evaluation_result['evaluation_time'] = evaluation_time
                    evaluation_result['judge_response_full'] = judge_text
                    return evaluation_result
                else:
                    print(f"Could not extract JSON from judge response", flush=True)
                    return {
                        'error': 'Failed to parse judge response',
                        'judge_response_full': judge_text,
                        'evaluation_time': evaluation_time
                    }
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}", flush=True)
                return {
                    'error': f'JSON decode error: {e}',
                    'judge_response_full': judge_text,
                    'evaluation_time': evaluation_time
                }
                
        except Exception as e:
            print(f"Judge evaluation failed: {e}", flush=True)
            return {
                'error': str(e),
                'evaluation_time': 0
            }
    
    def find_baseline_response(self, results: List[Dict], difficulty: str, question_num: int) -> Optional[str]:
        """Find the baseline response (A1 + Gemini Pro 2.5) for comparison"""
        for result in results:
            if (result['architecture'] == 'A1' and 
                result['model'] == 'gemini-2.5-pro' and
                result['difficulty'] == difficulty and
                result['question_num'] == question_num and
                result['success']):
                return result['response']
        return None
    
    def evaluate_all_responses_quality(self, results: List[Dict]):
        """Evaluate quality of all successful responses using LLM judge"""
        print("Starting quality evaluation phase...", flush=True)
        
        # Group results by difficulty and question for easier processing
        questions_by_difficulty = {}
        for result in results:
            if result['success']:  # Only evaluate successful responses
                difficulty = result['difficulty']
                question_num = result['question_num']
                
                if difficulty not in questions_by_difficulty:
                    questions_by_difficulty[difficulty] = {}
                if question_num not in questions_by_difficulty[difficulty]:
                    questions_by_difficulty[difficulty][question_num] = {
                        'prompt': result['prompt'],
                        'responses': []
                    }
                
                questions_by_difficulty[difficulty][question_num]['responses'].append(result)
        
        all_evaluations = []
        
        # Process each difficulty level
        for difficulty, questions in questions_by_difficulty.items():
            print(f"\n{'='*80}")
            print(f"EVALUATING {difficulty.upper()} QUESTIONS")
            print(f"{'='*80}")
            
            for question_num, question_data in questions.items():
                prompt = question_data['prompt']
                responses = question_data['responses']
                
                print(f"\nEvaluating Question {question_num} ({difficulty}): {prompt[:70]}...")
                
                # Find baseline response (A1 + Gemini Pro 2.5)
                baseline_response = self.find_baseline_response(results, difficulty, question_num)
                
                if not baseline_response:
                    print(f"No baseline response found for Q{question_num} ({difficulty}), skipping quality evaluation")
                    continue
                
                # Evaluate all other responses against baseline
                for response_result in responses:
                    arch = response_result['architecture']
                    model = response_result['model']
                    
                    # Skip baseline (don't compare to itself)
                    if arch == 'A1' and model == 'gemini-2.5-pro':
                        continue
                    
                    test_response = response_result['response']
                    
                    # Perform quality evaluation
                    evaluation = self.evaluate_response_quality(
                        prompt, baseline_response, test_response,
                        difficulty, arch, model
                    )
                    
                    if evaluation:
                        evaluation_record = {
                            'architecture': arch,
                            'model': model,
                            'difficulty': difficulty,
                            'question_num': question_num,
                            'prompt': prompt,
                            'baseline_response': baseline_response,
                            'test_response': test_response,
                            'evaluation': evaluation,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        all_evaluations.append(evaluation_record)
                        self.log_quality_evaluation(evaluation_record)
                        
                        # Add delay between judge evaluations
                        time.sleep(15)
        
        # Generate quality analysis summaries
        if all_evaluations:
            self.generate_quality_summaries(all_evaluations)
        else:
            print("No quality evaluations were performed")
    
    def log_quality_evaluation(self, evaluation_record: Dict):
        """Log quality evaluation to appropriate directory"""
        arch = evaluation_record['architecture']
        model = evaluation_record['model']
        difficulty = evaluation_record['difficulty']
        question_num = evaluation_record['question_num']
        
        # Get the specific directory for this architecture-model combination
        result_dir = self.directories[arch][model]
        
        # Create quality evaluation file
        eval_file = result_dir / "quality_evaluations" / f"Q{question_num}_{difficulty}_quality_eval.json"
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_record, f, indent=2, ensure_ascii=False)
        
        # Also create a human-readable version
        readable_file = result_dir / "quality_evaluations" / f"Q{question_num}_{difficulty}_quality_eval.txt"
        
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(f"QUALITY EVALUATION REPORT\n")
            f.write(f"{'='*80}\n")
            f.write(f"ARCHITECTURE: {arch}\n")
            f.write(f"MODEL: {model}\n")
            f.write(f"DIFFICULTY: {difficulty}\n")
            f.write(f"QUESTION: {question_num}\n")
            f.write(f"TIMESTAMP: {evaluation_record['timestamp']}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"PROMPT:\n{evaluation_record['prompt']}\n\n")
            f.write(f"{'='*80}\n")
            
            evaluation = evaluation_record['evaluation']
            
            if 'error' not in evaluation:
                f.write(f"QUALITY SCORES:\n")
                f.write(f"- Factual Accuracy: {evaluation.get('factual_accuracy', {}).get('score', 'N/A')}/10\n")
                f.write(f"- Completeness: {evaluation.get('completeness', {}).get('score', 'N/A')}/10\n")
                f.write(f"- Scientific Rigor: {evaluation.get('scientific_rigor', {}).get('score', 'N/A')}/10\n")
                f.write(f"- Clarity: {evaluation.get('clarity', {}).get('score', 'N/A')}/10\n")
                f.write(f"- Hallucination Detection: {evaluation.get('hallucination_detection', {}).get('score', 'N/A')}/10\n")
                f.write(f"- Overall Quality Score: {evaluation.get('overall_quality_score', 'N/A')}/10\n\n")
                
                f.write(f"ASSESSMENT:\n")
                f.write(f"- Hallucination Risk: {evaluation.get('hallucination_risk', 'N/A')}\n")
                f.write(f"- Comparison Verdict: {evaluation.get('comparison_verdict', 'N/A')}\n\n")
                
                if 'key_differences' in evaluation:
                    f.write(f"KEY DIFFERENCES:\n")
                    for i, diff in enumerate(evaluation['key_differences'], 1):
                        f.write(f"{i}. {diff}\n")
                    f.write("\n")
                
                if 'improvement_suggestions' in evaluation:
                    f.write(f"IMPROVEMENT SUGGESTIONS:\n")
                    for i, suggestion in enumerate(evaluation['improvement_suggestions'], 1):
                        f.write(f"{i}. {suggestion}\n")
                    f.write("\n")
            else:
                f.write(f"EVALUATION ERROR: {evaluation['error']}\n\n")
            
            f.write(f"{'='*80}\n")
            f.write(f"EVALUATION TIME: {evaluation.get('evaluation_time', 0):.2f}s\n")
    
    def test_architecture_model_on_prompt(self, architecture: Dict, model: Dict, prompt: str, difficulty: str, question_num: int) -> Dict:
        """Test a single architecture-model combination on a single prompt"""
        architecture_name = architecture['name']
        model_name = model['name']
        combo_name = f"{architecture_name}_{model_name}"
        
        result = {
            'architecture': architecture_name,
            'architecture_description': architecture['description'],
            'model': model_name,
            'combo_name': combo_name,
            'difficulty': difficulty,
            'question_num': question_num,
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'response': None,
            'error': None,
            'execution_time': 0,
            'retry_count': 0,
            'api_rotations': 0
        }
        
        max_retries = 3 if model_name == 'gemini-2.5-pro' else 1
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if retry_count == 0:
                    print(f"\n{'='*80}")
                    print(f"Testing {architecture_name} ({architecture['description']}) + {model_name}")
                    print(f"Question {question_num} ({difficulty}): {prompt[:100]}...")
                    print(f"{'='*80}")
                else:
                    print(f"\nRetry {retry_count}/{max_retries} for {combo_name}")
                
                # Rate limiting
                self.wait_for_rate_limit(model_name, model.get('rate_limit'))
                
                # Create agent and run test
                start_time = time.time()
                agent = self.create_agent(architecture, model['config'])
                
                print(f"Agent {architecture_name} created successfully with {model_name}")
                response = agent.go(prompt)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Record successful request for Gemini models
                if model_name == 'gemini-2.5-pro':
                    self._record_successful_gemini_request()
                
                result.update({
                    'success': True,
                    'response': str(response),
                    'execution_time': execution_time,
                    'retry_count': retry_count
                })
                
                print(f"SUCCESS: {combo_name} completed in {execution_time:.2f}s (attempt {retry_count + 1})")
                break
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                error_msg = str(e)
                
                print(f"ERROR: {combo_name} failed after {execution_time:.2f}s (attempt {retry_count + 1})")
                print(f"Error: {error_msg}")
                
                # Handle API errors for Gemini models
                should_retry = False
                if model_name == 'gemini-2.5-pro' and retry_count < max_retries:
                    # Check if this is a rate limit error (429)
                    response_code = None
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        response_code = 429
                    
                    should_retry = self._handle_gemini_api_error(e, response_code)
                    if should_retry:
                        result['api_rotations'] += 1
                        print(f"Will retry with rotated API key...")
                        time.sleep(self.gemini_rotation_manager.retry_delay_seconds if self.gemini_rotation_manager else 5)
                
                if should_retry:
                    retry_count += 1
                    continue
                else:
                    result.update({
                        'success': False,
                        'error': error_msg,
                        'execution_time': execution_time,
                        'retry_count': retry_count
                    })
                    
                    if retry_count < max_retries:
                        print(f"Non-retryable error or no rotation available")
                    
                    traceback.print_exc()
                    break
        
        return result
    
    def log_result(self, result: Dict):
        """Log result to architecture-model-specific directory"""
        architecture_name = result['architecture']
        model_name = result['model']
        
        # Get the specific directory for this architecture-model combination
        result_dir = self.directories[architecture_name][model_name]
        
        # Create individual test file
        test_file = result_dir / "individual_tests" / f"Q{result['question_num']}_{result['difficulty']}_test.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(f"ARCHITECTURE: {result['architecture']} ({result['architecture_description']})\n")
            f.write(f"MODEL: {result['model']}\n")
            f.write(f"DIFFICULTY: {result['difficulty']}\n") 
            f.write(f"QUESTION: {result['question_num']}\n")
            f.write(f"TIMESTAMP: {result['timestamp']}\n")
            f.write(f"SUCCESS: {result['success']}\n")
            f.write(f"EXECUTION TIME: {result['execution_time']:.2f}s\n")
            f.write(f"RETRY COUNT: {result.get('retry_count', 0)}\n")
            f.write(f"API ROTATIONS: {result.get('api_rotations', 0)}\n")
            f.write(f"{'='*80}\n")
            f.write(f"PROMPT:\n{result['prompt']}\n")
            f.write(f"{'='*80}\n")
            
            if result['success']:
                f.write(f"RESPONSE:\n{result['response']}\n")
            else:
                f.write(f"ERROR:\n{result['error']}\n")
        
        # Also append to the main log file for this combination
        main_log_file = result_dir / "logs" / f"{architecture_name}_{model_name}_all_tests.txt"
        
        with open(main_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"Q{result['question_num']} ({result['difficulty']}) - {result['timestamp']}\n")
            f.write(f"Success: {result['success']} | Time: {result['execution_time']:.2f}s\n")
            f.write(f"Prompt: {result['prompt'][:100]}...\n")
            f.write(f"{'='*100}\n")
            
            if result['success']:
                # Only store first 500 chars of response in main log to keep it manageable
                response_preview = str(result['response'])[:500]
                if len(str(result['response'])) > 500:
                    response_preview += "... (see individual test file for full response)"
                f.write(f"Response: {response_preview}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            
            f.write(f"\n")
    
    def run_comprehensive_test(self):
        """Run all prompts on all architecture-model combinations, question by question"""
        total_combinations = len(self.architectures) * len(self.models)
        total_prompts = sum(len(prompts) for prompts in self.prompts.values())
        total_tests = total_combinations * total_prompts
        
        print(f"Starting comprehensive architecture + model test at {datetime.now()}")
        print(f"Testing {len(self.architectures)} architectures × {len(self.models)} models = {total_combinations} combinations")
        print(f"On {total_prompts} prompts = {total_tests} total tests")
        print(f"Results will be organized in: {self.base_log_dir}")
        
        # Clear existing log files
        for arch_name in self.directories:
            for model_name in self.directories[arch_name]:
                result_dir = self.directories[arch_name][model_name]
                # Clear main log files
                main_log_file = result_dir / "logs" / f"{arch_name}_{model_name}_all_tests.txt"
                if main_log_file.exists():
                    main_log_file.unlink()
                # Clear individual test files
                for test_file in (result_dir / "individual_tests").glob("*.txt"):
                    test_file.unlink()
        
        all_results = []
        
        # Process each difficulty level
        for difficulty in ['easy']:
            if not self.prompts[difficulty]:
                continue
                
            print(f"\n{'#'*100}")
            print(f"TESTING {difficulty.upper()} QUESTIONS")
            print(f"{'#'*100}")
            
            # Process each question
            for i, prompt in enumerate(self.prompts[difficulty], 1):
                print(f"\n{'*'*80}")
                print(f"QUESTION {i} ({difficulty}): {prompt[:70]}...")
                print(f"{'*'*80}")
                
                # Test this question on all architecture-model combinations
                for architecture in self.architectures:
                    for model in self.models:
                        result = self.test_architecture_model_on_prompt(
                            architecture, model, prompt, difficulty, i
                        )
                        self.log_result(result)
                        all_results.append(result)
                        
                        # Small delay between tests
                        time.sleep(20)
                        
                    ###TODO
                    break
        
        # Generate summaries
        self.generate_summaries(all_results)
        
        # Perform quality evaluation with LLM judge
        print(f"\n{'#'*100}")
        print(f"STARTING QUALITY EVALUATION WITH LLM JUDGE")
        print(f"{'#'*100}")
        self.evaluate_all_responses_quality(all_results)
        
        print(f"\nComprehensive test completed at {datetime.now()}")
        print(f"Check {self.base_log_dir} for organized results!")
        print(f"Quality evaluations available in quality_evaluations/ directories")
        
        # Print final API rotation statistics
        if self.gemini_rotation_manager:
            print(f"\n{'#'*100}")
            print(f"FINAL GEMINI API ROTATION STATISTICS")
            print(f"{'#'*100}")
            self.gemini_rotation_manager.print_status()
    
    def generate_summaries(self, results: List[Dict]):
        """Generate comprehensive summaries in organized directories"""
        
        # Overall summary
        overall_summary_file = self.comparison_dir / "overall" / "comprehensive_summary.txt"
        self.write_overall_summary(results, overall_summary_file)
        
        # Per-architecture summaries
        arch_results = {}
        for result in results:
            arch = result['architecture']
            if arch not in arch_results:
                arch_results[arch] = []
            arch_results[arch].append(result)
        
        for arch_name, arch_data in arch_results.items():
            arch_summary_file = self.comparison_dir / "by_architecture" / f"{arch_name}_summary.txt"
            self.write_architecture_summary(arch_name, arch_data, arch_summary_file)
            
            # Also write to the architecture's own directory
            arch_dir_summary = self.base_log_dir / arch_name / f"{arch_name}_overall_summary.txt"
            self.write_architecture_summary(arch_name, arch_data, arch_dir_summary)
        
        # Per-model summaries
        model_results = {}
        for result in results:
            model = result['model']
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(result)
        
        for model_name, model_data in model_results.items():
            model_summary_file = self.comparison_dir / "by_model" / f"{model_name}_summary.txt"
            self.write_model_summary(model_name, model_data, model_summary_file)
        
        # Per-combination summaries (in their respective directories)
        combo_results = {}
        for result in results:
            arch = result['architecture']
            model = result['model']
            combo_key = f"{arch}_{model}"
            if combo_key not in combo_results:
                combo_results[combo_key] = []
            combo_results[combo_key].append(result)
        
        for combo_key, combo_data in combo_results.items():
            arch_name, model_name = combo_key.split('_', 1)
            result_dir = self.directories[arch_name][model_name]
            combo_summary_file = result_dir / "summaries" / f"{combo_key}_summary.txt"
            self.write_combination_summary(arch_name, model_name, combo_data, combo_summary_file)
        
        # Difficulty comparison
        difficulty_summary_file = self.comparison_dir / "by_difficulty" / "difficulty_comparison.txt"
        self.write_difficulty_summary(results, difficulty_summary_file)
        
        print(f"All summaries generated in {self.comparison_dir}")
    
    def write_overall_summary(self, results: List[Dict], summary_file: Path):
        """Write the overall comprehensive summary"""
        with open(summary_file, 'w') as f:
            f.write(f"COMPREHENSIVE ARCHITECTURE + MODEL TEST SUMMARY\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            
            # Overall stats
            total_tests = len(results)
            successful_tests = sum(1 for r in results if r['success'])
            
            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Successful: {successful_tests}\n")
            f.write(f"Failed: {total_tests - successful_tests}\n")
            f.write(f"Success Rate: {successful_tests/total_tests*100:.1f}%\n\n")
            
            # Architecture vs Model success rate matrix
            f.write(f"ARCHITECTURE vs MODEL SUCCESS RATE MATRIX:\n")
            f.write(f"{'Architecture':<15}")
            for model in self.models:
                f.write(f"{model['name']:<30}")
            f.write("\n")
            f.write("-" * (15 + 30 * len(self.models)) + "\n")
            
            for arch in self.architectures:
                f.write(f"{arch['name']:<15}")
                for model in self.models:
                    combo_results = [r for r in results if r['architecture'] == arch['name'] and r['model'] == model['name']]
                    if combo_results:
                        success_rate = sum(1 for r in combo_results if r['success']) / len(combo_results) * 100
                        f.write(f"{success_rate:>8.1f}%{'':<21}")
                    else:
                        f.write(f"{'N/A':<30}")
                f.write("\n")
    
    def write_architecture_summary(self, arch_name: str, arch_results: List[Dict], summary_file: Path):
        """Write summary for a specific architecture"""
        with open(summary_file, 'w') as f:
            f.write(f"ARCHITECTURE SUMMARY: {arch_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            
            total_tests = len(arch_results)
            successful_tests = sum(1 for r in arch_results if r['success'])
            avg_time = sum(r['execution_time'] for r in arch_results) / total_tests
            
            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Successful: {successful_tests}\n")
            f.write(f"Success Rate: {successful_tests/total_tests*100:.1f}%\n")
            f.write(f"Average Execution Time: {avg_time:.2f}s\n\n")
            
            # Per-model breakdown
            f.write(f"PER-MODEL BREAKDOWN:\n")
            model_stats = {}
            for result in arch_results:
                model = result['model']
                if model not in model_stats:
                    model_stats[model] = {'total': 0, 'success': 0, 'times': []}
                model_stats[model]['total'] += 1
                if result['success']:
                    model_stats[model]['success'] += 1
                model_stats[model]['times'].append(result['execution_time'])
            
            for model, stats in model_stats.items():
                success_rate = stats['success'] / stats['total'] * 100
                avg_time = sum(stats['times']) / len(stats['times'])
                f.write(f"\n{model}:\n")
                f.write(f"  Tests: {stats['total']}\n")
                f.write(f"  Success: {stats['success']}\n")
                f.write(f"  Success Rate: {success_rate:.1f}%\n")
                f.write(f"  Avg Time: {avg_time:.2f}s\n")
    
    def write_model_summary(self, model_name: str, model_results: List[Dict], summary_file: Path):
        """Write summary for a specific model"""
        with open(summary_file, 'w') as f:
            f.write(f"MODEL SUMMARY: {model_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            
            total_tests = len(model_results)
            successful_tests = sum(1 for r in model_results if r['success'])
            avg_time = sum(r['execution_time'] for r in model_results) / total_tests
            
            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Successful: {successful_tests}\n")
            f.write(f"Success Rate: {successful_tests/total_tests*100:.1f}%\n")
            f.write(f"Average Execution Time: {avg_time:.2f}s\n\n")
            
            # Per-architecture breakdown
            f.write(f"PER-ARCHITECTURE BREAKDOWN:\n")
            arch_stats = {}
            for result in model_results:
                arch = result['architecture']
                if arch not in arch_stats:
                    arch_stats[arch] = {'total': 0, 'success': 0, 'times': []}
                arch_stats[arch]['total'] += 1
                if result['success']:
                    arch_stats[arch]['success'] += 1
                arch_stats[arch]['times'].append(result['execution_time'])
            
            for arch, stats in arch_stats.items():
                success_rate = stats['success'] / stats['total'] * 100
                avg_time = sum(stats['times']) / len(stats['times'])
                f.write(f"\n{arch}:\n")
                f.write(f"  Tests: {stats['total']}\n")
                f.write(f"  Success: {stats['success']}\n")
                f.write(f"  Success Rate: {success_rate:.1f}%\n")
                f.write(f"  Avg Time: {avg_time:.2f}s\n")
    
    def write_combination_summary(self, arch_name: str, model_name: str, combo_results: List[Dict], summary_file: Path):
        """Write summary for a specific architecture-model combination"""
        with open(summary_file, 'w') as f:
            f.write(f"COMBINATION SUMMARY: {arch_name} + {model_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            
            total_tests = len(combo_results)
            successful_tests = sum(1 for r in combo_results if r['success'])
            avg_time = sum(r['execution_time'] for r in combo_results) / total_tests
            
            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Successful: {successful_tests}\n")
            f.write(f"Success Rate: {successful_tests/total_tests*100:.1f}%\n")
            f.write(f"Average Execution Time: {avg_time:.2f}s\n\n")
            
            # Per-difficulty breakdown
            f.write(f"PER-DIFFICULTY BREAKDOWN:\n")
            diff_stats = {}
            for result in combo_results:
                diff = result['difficulty']
                if diff not in diff_stats:
                    diff_stats[diff] = {'total': 0, 'success': 0, 'times': []}
                diff_stats[diff]['total'] += 1
                if result['success']:
                    diff_stats[diff]['success'] += 1
                diff_stats[diff]['times'].append(result['execution_time'])
            
            for diff, stats in diff_stats.items():
                success_rate = stats['success'] / stats['total'] * 100
                avg_time = sum(stats['times']) / len(stats['times'])
                f.write(f"\n{diff.upper()}:\n")
                f.write(f"  Tests: {stats['total']}\n")
                f.write(f"  Success: {stats['success']}\n")
                f.write(f"  Success Rate: {success_rate:.1f}%\n")
                f.write(f"  Avg Time: {avg_time:.2f}s\n")
            
            # List of failed tests for debugging
            failed_tests = [r for r in combo_results if not r['success']]
            if failed_tests:
                f.write(f"\nFAILED TESTS:\n")
                for i, result in enumerate(failed_tests, 1):
                    f.write(f"{i}. Q{result['question_num']} ({result['difficulty']}): {result['error'][:100]}...\n")
    
    def write_difficulty_summary(self, results: List[Dict], summary_file: Path):
        """Write difficulty comparison summary"""
        with open(summary_file, 'w') as f:
            f.write(f"DIFFICULTY COMPARISON SUMMARY\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            
            # Group by difficulty
            difficulty_stats = {}
            for result in results:
                diff = result['difficulty']
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {'total': 0, 'success': 0, 'times': []}
                difficulty_stats[diff]['total'] += 1
                if result['success']:
                    difficulty_stats[diff]['success'] += 1
                difficulty_stats[diff]['times'].append(result['execution_time'])
            
            f.write(f"OVERALL DIFFICULTY PERFORMANCE:\n")
            for diff, stats in difficulty_stats.items():
                success_rate = stats['success'] / stats['total'] * 100
                avg_time = sum(stats['times']) / len(stats['times'])
                f.write(f"\n{diff.upper()}:\n")
                f.write(f"  Tests: {stats['total']}\n")
                f.write(f"  Success: {stats['success']}\n")
                f.write(f"  Success Rate: {success_rate:.1f}%\n")
                f.write(f"  Avg Time: {avg_time:.2f}s\n")
    
    def generate_quality_summaries(self, evaluations: List[Dict]):
        """Generate quality analysis summaries"""
        print("Generating quality analysis summaries...", flush=True)
        
        # Overall quality summary
        overall_quality_file = self.comparison_dir / "quality_analysis" / "overall_quality_analysis.txt"
        self.write_overall_quality_summary(evaluations, overall_quality_file)
        
        # Per-architecture quality summaries
        arch_evaluations = {}
        for evaluation in evaluations:
            arch = evaluation['architecture']
            if arch not in arch_evaluations:
                arch_evaluations[arch] = []
            arch_evaluations[arch].append(evaluation)
        
        for arch_name, arch_evals in arch_evaluations.items():
            arch_quality_file = self.comparison_dir / "quality_analysis" / f"{arch_name}_quality_analysis.txt"
            self.write_architecture_quality_summary(arch_name, arch_evals, arch_quality_file)
        
        # Per-difficulty quality summaries
        diff_evaluations = {}
        for evaluation in evaluations:
            diff = evaluation['difficulty']
            if diff not in diff_evaluations:
                diff_evaluations[diff] = []
            diff_evaluations[diff].append(evaluation)
        
        for diff_name, diff_evals in diff_evaluations.items():
            diff_quality_file = self.comparison_dir / "quality_analysis" / f"{diff_name}_difficulty_quality_analysis.txt"
            self.write_difficulty_quality_summary(diff_name, diff_evals, diff_quality_file)
        
        print(f"Quality analysis summaries generated in {self.comparison_dir / 'quality_analysis'}")
    
    def write_overall_quality_summary(self, evaluations: List[Dict], summary_file: Path):
        """Write overall quality analysis summary"""
        with open(summary_file, 'w') as f:
            f.write(f"OVERALL QUALITY ANALYSIS SUMMARY\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Baseline: A1 + Gemini Pro 2.5\n")
            f.write(f"{'='*80}\n\n")
            
            total_evaluations = len(evaluations)
            f.write(f"TOTAL EVALUATIONS: {total_evaluations}\n\n")
            
            # Calculate average scores across all evaluations
            valid_evals = [e for e in evaluations if 'error' not in e['evaluation']]
            if valid_evals:
                avg_scores = {}
                criteria = ['factual_accuracy', 'completeness', 'scientific_rigor', 'clarity', 'hallucination_detection']
                
                for criterion in criteria:
                    scores = [e['evaluation'][criterion]['score'] for e in valid_evals if criterion in e['evaluation']]
                    avg_scores[criterion] = sum(scores) / len(scores) if scores else 0
                
                overall_scores = [e['evaluation']['overall_quality_score'] for e in valid_evals if 'overall_quality_score' in e['evaluation']]
                avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
                
                f.write(f"AVERAGE QUALITY SCORES (vs A1+Gemini baseline):\n")
                f.write(f"- Factual Accuracy: {avg_scores.get('factual_accuracy', 0):.1f}/10\n")
                f.write(f"- Completeness: {avg_scores.get('completeness', 0):.1f}/10\n")
                f.write(f"- Scientific Rigor: {avg_scores.get('scientific_rigor', 0):.1f}/10\n")
                f.write(f"- Clarity: {avg_scores.get('clarity', 0):.1f}/10\n")
                f.write(f"- Hallucination Detection: {avg_scores.get('hallucination_detection', 0):.1f}/10\n")
                f.write(f"- Overall Quality Score: {avg_overall:.1f}/10\n\n")
                
                # Hallucination risk analysis
                risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
                for eval_record in valid_evals:
                    risk = eval_record['evaluation'].get('hallucination_risk', 'UNKNOWN')
                    if risk in risk_counts:
                        risk_counts[risk] += 1
                
                f.write(f"HALLUCINATION RISK DISTRIBUTION:\n")
                for risk, count in risk_counts.items():
                    percentage = count / len(valid_evals) * 100 if valid_evals else 0
                    f.write(f"- {risk}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Comparison verdicts
                verdict_counts = {'BETTER': 0, 'EQUAL': 0, 'WORSE': 0}
                for eval_record in valid_evals:
                    verdict = eval_record['evaluation'].get('comparison_verdict', 'UNKNOWN')
                    if verdict in verdict_counts:
                        verdict_counts[verdict] += 1
                
                f.write(f"COMPARISON VERDICTS (vs A1+Gemini baseline):\n")
                for verdict, count in verdict_counts.items():
                    percentage = count / len(valid_evals) * 100 if valid_evals else 0
                    f.write(f"- {verdict}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Architecture performance matrix
            f.write(f"ARCHITECTURE QUALITY MATRIX:\n")
            f.write(f"{'Architecture':<15}{'Model':<25}{'Avg Quality':<15}{'Halluc Risk':<15}{'Verdict'}\n")
            f.write("-" * 80 + "\n")
            
            # Group by architecture and model
            combo_stats = {}
            for eval_record in valid_evals:
                arch = eval_record['architecture']
                model = eval_record['model']
                combo = f"{arch}+{model}"
                
                if combo not in combo_stats:
                    combo_stats[combo] = {
                        'scores': [],
                        'risks': [],
                        'verdicts': [],
                        'arch': arch,
                        'model': model
                    }
                
                if 'overall_quality_score' in eval_record['evaluation']:
                    combo_stats[combo]['scores'].append(eval_record['evaluation']['overall_quality_score'])
                if 'hallucination_risk' in eval_record['evaluation']:
                    combo_stats[combo]['risks'].append(eval_record['evaluation']['hallucination_risk'])
                if 'comparison_verdict' in eval_record['evaluation']:
                    combo_stats[combo]['verdicts'].append(eval_record['evaluation']['comparison_verdict'])
            
            for combo, stats in combo_stats.items():
                avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
                
                # Most common risk and verdict
                most_common_risk = max(set(stats['risks']), key=stats['risks'].count) if stats['risks'] else 'N/A'
                most_common_verdict = max(set(stats['verdicts']), key=stats['verdicts'].count) if stats['verdicts'] else 'N/A'
                
                f.write(f"{stats['arch']:<15}{stats['model']:<25}{avg_score:<15.1f}{most_common_risk:<15}{most_common_verdict}\n")
    
    def write_architecture_quality_summary(self, arch_name: str, arch_evaluations: List[Dict], summary_file: Path):
        """Write quality summary for a specific architecture"""
        with open(summary_file, 'w') as f:
            f.write(f"ARCHITECTURE QUALITY ANALYSIS: {arch_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Baseline: A1 + Gemini Pro 2.5\n")
            f.write(f"{'='*80}\n\n")
            
            valid_evals = [e for e in arch_evaluations if 'error' not in e['evaluation']]
            
            if valid_evals:
                # Calculate average scores for this architecture
                criteria = ['factual_accuracy', 'completeness', 'scientific_rigor', 'clarity', 'hallucination_detection']
                avg_scores = {}
                
                for criterion in criteria:
                    scores = [e['evaluation'][criterion]['score'] for e in valid_evals if criterion in e['evaluation']]
                    avg_scores[criterion] = sum(scores) / len(scores) if scores else 0
                
                overall_scores = [e['evaluation']['overall_quality_score'] for e in valid_evals if 'overall_quality_score' in e['evaluation']]
                avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
                
                f.write(f"AVERAGE QUALITY SCORES FOR {arch_name}:\n")
                f.write(f"- Factual Accuracy: {avg_scores.get('factual_accuracy', 0):.1f}/10\n")
                f.write(f"- Completeness: {avg_scores.get('completeness', 0):.1f}/10\n")
                f.write(f"- Scientific Rigor: {avg_scores.get('scientific_rigor', 0):.1f}/10\n")
                f.write(f"- Clarity: {avg_scores.get('clarity', 0):.1f}/10\n")
                f.write(f"- Hallucination Detection: {avg_scores.get('hallucination_detection', 0):.1f}/10\n")
                f.write(f"- Overall Quality Score: {avg_overall:.1f}/10\n\n")
                
                # Per-model breakdown
                model_stats = {}
                for eval_record in valid_evals:
                    model = eval_record['model']
                    if model not in model_stats:
                        model_stats[model] = {'scores': [], 'risks': [], 'verdicts': []}
                    
                    if 'overall_quality_score' in eval_record['evaluation']:
                        model_stats[model]['scores'].append(eval_record['evaluation']['overall_quality_score'])
                    if 'hallucination_risk' in eval_record['evaluation']:
                        model_stats[model]['risks'].append(eval_record['evaluation']['hallucination_risk'])
                    if 'comparison_verdict' in eval_record['evaluation']:
                        model_stats[model]['verdicts'].append(eval_record['evaluation']['comparison_verdict'])
                
                f.write(f"PER-MODEL BREAKDOWN:\n")
                for model, stats in model_stats.items():
                    avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
                    most_common_risk = max(set(stats['risks']), key=stats['risks'].count) if stats['risks'] else 'N/A'
                    most_common_verdict = max(set(stats['verdicts']), key=stats['verdicts'].count) if stats['verdicts'] else 'N/A'
                    
                    f.write(f"\n{model}:\n")
                    f.write(f"  Avg Quality Score: {avg_score:.1f}/10\n")
                    f.write(f"  Most Common Risk: {most_common_risk}\n")
                    f.write(f"  Most Common Verdict: {most_common_verdict}\n")
    
    def write_difficulty_quality_summary(self, difficulty: str, diff_evaluations: List[Dict], summary_file: Path):
        """Write quality summary for a specific difficulty level"""
        with open(summary_file, 'w') as f:
            f.write(f"DIFFICULTY QUALITY ANALYSIS: {difficulty.upper()}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Baseline: A1 + Gemini Pro 2.5\n")
            f.write(f"{'='*80}\n\n")
            
            valid_evals = [e for e in diff_evaluations if 'error' not in e['evaluation']]
            
            if valid_evals:
                # Calculate average scores for this difficulty
                criteria = ['factual_accuracy', 'completeness', 'scientific_rigor', 'clarity', 'hallucination_detection']
                avg_scores = {}
                
                for criterion in criteria:
                    scores = [e['evaluation'][criterion]['score'] for e in valid_evals if criterion in e['evaluation']]
                    avg_scores[criterion] = sum(scores) / len(scores) if scores else 0
                
                overall_scores = [e['evaluation']['overall_quality_score'] for e in valid_evals if 'overall_quality_score' in e['evaluation']]
                avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
                
                f.write(f"AVERAGE QUALITY SCORES FOR {difficulty.upper()} QUESTIONS:\n")
                f.write(f"- Factual Accuracy: {avg_scores.get('factual_accuracy', 0):.1f}/10\n")
                f.write(f"- Completeness: {avg_scores.get('completeness', 0):.1f}/10\n")
                f.write(f"- Scientific Rigor: {avg_scores.get('scientific_rigor', 0):.1f}/10\n")
                f.write(f"- Clarity: {avg_scores.get('clarity', 0):.1f}/10\n")
                f.write(f"- Hallucination Detection: {avg_scores.get('hallucination_detection', 0):.1f}/10\n")
                f.write(f"- Overall Quality Score: {avg_overall:.1f}/10\n\n")
                
                # Per-architecture breakdown for this difficulty
                arch_stats = {}
                for eval_record in valid_evals:
                    arch = eval_record['architecture']
                    if arch not in arch_stats:
                        arch_stats[arch] = {'scores': [], 'risks': [], 'verdicts': []}
                    
                    if 'overall_quality_score' in eval_record['evaluation']:
                        arch_stats[arch]['scores'].append(eval_record['evaluation']['overall_quality_score'])
                    if 'hallucination_risk' in eval_record['evaluation']:
                        arch_stats[arch]['risks'].append(eval_record['evaluation']['hallucination_risk'])
                    if 'comparison_verdict' in eval_record['evaluation']:
                        arch_stats[arch]['verdicts'].append(eval_record['evaluation']['comparison_verdict'])
                
                f.write(f"PER-ARCHITECTURE BREAKDOWN FOR {difficulty.upper()}:\n")
                for arch, stats in arch_stats.items():
                    avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
                    most_common_risk = max(set(stats['risks']), key=stats['risks'].count) if stats['risks'] else 'N/A'
                    most_common_verdict = max(set(stats['verdicts']), key=stats['verdicts'].count) if stats['verdicts'] else 'N/A'
                    
                    f.write(f"\n{arch}:\n")
                    f.write(f"  Avg Quality Score: {avg_score:.1f}/10\n")
                    f.write(f"  Most Common Risk: {most_common_risk}\n")
                    f.write(f"  Most Common Verdict: {most_common_verdict}\n")


def main():
    """Main entry point"""
    print("=" * 100)
    print("COMPREHENSIVE ARCHITECTURE + MODEL TESTING SCRIPT")
    print("Testing A1, E1, F1 architectures across multiple models")
    print("Includes LLM-as-Judge quality evaluation and hallucination detection")
    print("Results organized by architecture and model for easy comparison")
    print("=" * 100)
    
    tester = ModelTester()
    tester.run_comprehensive_test()
    
    print(f"\nTest completed! Check the {tester.base_log_dir} directory for organized results:")
    print(f"  - Individual test results: {tester.base_log_dir}/[ARCHITECTURE]/[MODEL]/individual_tests/")
    print(f"  - Quality evaluations: {tester.base_log_dir}/[ARCHITECTURE]/[MODEL]/quality_evaluations/")
    print(f"  - Architecture summaries: {tester.base_log_dir}/[ARCHITECTURE]/[ARCHITECTURE]_overall_summary.txt")
    print(f"  - Combination summaries: {tester.base_log_dir}/[ARCHITECTURE]/[MODEL]/summaries/")
    print(f"  - Comparison summaries: {tester.base_log_dir}/comparisons/")
    print(f"  - Quality analysis: {tester.base_log_dir}/comparisons/quality_analysis/")


if __name__ == "__main__":
    main() 