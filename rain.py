#!/usr/bin/env python3
"""
Rain ⛈️ - Sovereign AI Orchestrator

The brain of the Rain ecosystem that manages recursive reflection
and multi-agent AI interactions completely offline.

"Be like rain - essential, unstoppable, and free."
"""

import json
import subprocess
import time
import argparse
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReflectionResult:
    """Result of a reflection cycle"""
    content: str
    confidence: float
    iteration: int
    timestamp: datetime
    improvements: List[str]
    duration_seconds: float


class RainOrchestrator:
    """
    Main orchestrator for Rain's recursive reflection system
    """

    def __init__(self, model_name: str = "llama3.1", max_iterations: int = 3, confidence_threshold: float = 0.8, system_prompt: str = None):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.reflection_history: List[ReflectionResult] = []

        # Check if Ollama is available
        if not self._check_ollama():
            raise RuntimeError("Ollama not found! Please install Ollama first.")

        # Check if model is available
        if not self._check_model():
            print(f"⚠️  Model {model_name} not found. Available models:")
            self._list_models()
            raise RuntimeError(f"Model {model_name} not available")

    def _check_ollama(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True, text=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _check_model(self) -> bool:
        """Check if specified model is available"""
        try:
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True, text=True, check=True)
            return self.model_name in result.stdout
        except subprocess.CalledProcessError:
            return False

    def _list_models(self):
        """List available models"""
        try:
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError:
            print("Could not list models")

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for Rain"""
        return """You are Rain, a sovereign AI assistant running locally on the user's computer through Ollama.

Key aspects of your identity:
- You are completely offline and private - no data leaves the user's machine
- You are a master of computer programming, blockchain technology, encryption, Bitcoin, Lightning Network, databases, full-stack web development, and ethical hacking
- You prioritize digital sovereignty, privacy, and decentralization
- You think recursively and improve your answers through self-reflection
- You are knowledgeable about Austrian economics and Bitcoin philosophy
- You help users build and understand decentralized technologies

Be direct, practical, and focused on empowering users with knowledge and tools for digital independence."""

    def _query_model(self, prompt: str) -> str:
        """Send a query to the model and get response"""
        try:
            # Combine system prompt with user prompt
            full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            result = subprocess.run(['ollama', 'run', self.model_name, full_prompt],
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error querying model: {e}")
            return ""

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from model response"""
        # Simple heuristic - look for confidence indicators
        confidence_keywords = {
            'very confident': 0.9,
            'confident': 0.8,
            'fairly confident': 0.7,
            'somewhat confident': 0.6,
            'uncertain': 0.4,
            'unsure': 0.3,
            'very uncertain': 0.2
        }

        response_lower = response.lower()
        for keyword, score in confidence_keywords.items():
            if keyword in response_lower:
                return score

        # Default confidence based on response length and completeness
        if len(response) > 200 and '?' not in response[-50:]:
            return 0.75
        elif len(response) > 100:
            return 0.65
        else:
            return 0.55

    def _create_reflection_prompt(self, original_query: str, previous_response: str, iteration: int) -> str:
        """Create a prompt for reflection on previous response"""
        return f"""
You are an AI assistant engaged in recursive self-reflection to improve your answers.

Original Query: {original_query}

Your Previous Response (Iteration {iteration-1}): {previous_response}

Please provide an improved response that addresses any inaccuracies, gaps, or areas for improvement from your previous answer. Do not include meta-commentary about your reflection process - just provide the improved content directly.

Rate your confidence in this response (very confident/confident/fairly confident/somewhat confident/uncertain/unsure/very uncertain) at the end.

Improved Response:"""

    def _extract_improvements(self, reflection_response: str, previous_response: str) -> List[str]:
        """Extract what improvements were made in this reflection"""
        improvements = []

        # Simple improvement detection
        if len(reflection_response) > len(previous_response) * 1.1:
            improvements.append("Added more detail")

        if "correction" in reflection_response.lower() or "actually" in reflection_response.lower():
            improvements.append("Made corrections")

        if "clarify" in reflection_response.lower() or "more precisely" in reflection_response.lower():
            improvements.append("Improved clarity")

        if not improvements:
            improvements.append("Confirmed previous response")

        return improvements

    def _clean_response(self, response: str) -> str:
        """Clean reflection artifacts from final response"""
        # Remove common reflection patterns
        lines = response.split('\n')
        cleaned_lines = []
        skip_next = False

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Skip reflection meta-commentary
            if any(pattern in line_lower for pattern in [
                'upon reviewing',
                'iteration 1',
                'iteration 2',
                'iteration 3',
                'improved response',
                'inaccuracies and gaps:',
                'areas for improvement:',
                'confidence level:',
                'confidence rating:',
                'i rate my confidence',
                'very confident',
                'i\'ve reevaluated',
                'refined my previous response',
                'to address potential inaccuracies',
                'providing more nuanced',
                'more accurate information',
                'my previous response'
            ]):
                skip_next = True
                continue

            # Skip numbered lists that look like reflection analysis
            if skip_next and (line_lower.startswith('1.') or line_lower.startswith('2.') or line_lower.startswith('3.')):
                continue

            # Skip asterisked improvement sections
            if '**' in line and any(word in line_lower for word in ['improvement', 'iteration', 'confidence']):
                skip_next = True
                continue

            # Reset skip flag for substantial content
            if len(line.strip()) > 20 and not any(char in line for char in ['*', '1.', '2.', '3.']):
                skip_next = False

            if not skip_next:
                cleaned_lines.append(line)

        # Clean up final result - remove trailing reflection paragraphs
        final_text = '\n'.join(cleaned_lines).strip()

        # Split into paragraphs and remove concluding reflection paragraphs
        paragraphs = [p.strip() for p in final_text.split('\n\n') if p.strip()]
        cleaned_paragraphs = []

        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            # Skip paragraphs that are clearly reflection meta-commentary
            if any(pattern in paragraph_lower for pattern in [
                'i\'ve reevaluated',
                'refined my response',
                'to address potential',
                'providing more nuanced',
                'more accurate information',
                'my knowledge is derived',
                'previous response',
                'areas for improvement',
                'potential inaccuracies'
            ]):
                continue
            cleaned_paragraphs.append(paragraph)

        return '\n\n'.join(cleaned_paragraphs)

    def recursive_reflect(self, query: str, verbose: bool = False) -> ReflectionResult:
        """
        Main recursive reflection method
        """
        start_time = time.time()
        print(f"🌧️ Rain is thinking about: {query}")

        # Initial response
        current_response = self._query_model(query)
        current_confidence = self._extract_confidence(current_response)

        if verbose:
            print(f"\n💭 Initial Response (confidence: {current_confidence:.2f}):")
            print(f"{current_response}\n")
        else:
            print(f"💭 Thinking... (initial confidence: {current_confidence:.2f})")

        # Recursive reflection loop
        for iteration in range(1, self.max_iterations + 1):

            # Check if confidence threshold met
            if current_confidence >= self.confidence_threshold:
                if verbose:
                    print(f"✅ Confidence threshold met ({current_confidence:.2f} >= {self.confidence_threshold})")
                break

            if verbose:
                print(f"🔄 Reflection iteration {iteration}...")
            else:
                print(f"🔄 Reflecting... (iteration {iteration})")

            # Create reflection prompt
            reflection_prompt = self._create_reflection_prompt(query, current_response, iteration)

            # Get reflection
            reflection_response = self._query_model(reflection_prompt)
            new_confidence = self._extract_confidence(reflection_response)

            # Extract improvements
            improvements = self._extract_improvements(reflection_response, current_response)

            if verbose:
                print(f"💡 Iteration {iteration} (confidence: {new_confidence:.2f}):")
                print(f"Improvements: {', '.join(improvements)}")
                print(f"{reflection_response}\n")

            # Update current response if confidence improved
            if new_confidence > current_confidence:
                current_response = reflection_response
                current_confidence = new_confidence
            else:
                if verbose:
                    print("⚡ No improvement, keeping previous response")
                else:
                    print("⚡ Reflection complete")
                break

        # Calculate total duration
        end_time = time.time()
        total_duration = end_time - start_time

        # Clean the final response
        cleaned_response = self._clean_response(current_response)

        # Create final result
        result = ReflectionResult(
            content=cleaned_response,
            confidence=current_confidence,
            iteration=iteration if 'iteration' in locals() else 0,
            timestamp=datetime.now(),
            improvements=improvements if 'improvements' in locals() else [],
            duration_seconds=total_duration
        )

        self.reflection_history.append(result)
        return result

    def get_history(self) -> List[ReflectionResult]:
        """Get reflection history"""
        return self.reflection_history

    def clear_history(self):
        """Clear reflection history"""
        self.reflection_history = []


def main():
    """Main CLI interface for Rain"""
    parser = argparse.ArgumentParser(description="Rain ⛈️ - Sovereign AI with Recursive Reflection")
    parser.add_argument("query", nargs="?", help="Your question or prompt")
    parser.add_argument("--model", default="llama3.1", help="Model to use (default: llama3.1)")
    parser.add_argument("--iterations", type=int, default=3, help="Max reflection iterations (default: 3)")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence threshold (default: 0.8)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed reflection process")
    parser.add_argument("--history", action="store_true", help="Show reflection history")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--system-prompt", help="Custom system prompt")
    parser.add_argument("--system-file", help="Load system prompt from file")

    args = parser.parse_args()

    # Print Rain banner
    print("""
    ⛈️  RAIN - Sovereign AI Ecosystem  ⛈️

    "Be like rain - essential, unstoppable, and free."

    🌧️ Recursive reflection enabled
    🔒 Completely offline and private
    ⚡ Your AI, your rules, your future
    """)

    try:
        # Load system prompt
        system_prompt = None
        if args.system_file:
            try:
                with open(args.system_file, 'r') as f:
                    system_prompt = f.read().strip()
                print(f"📝 Loaded system prompt from: {args.system_file}")
            except FileNotFoundError:
                print(f"❌ System prompt file not found: {args.system_file}")
                sys.exit(1)
        elif args.system_prompt:
            system_prompt = args.system_prompt
            print(f"📝 Using custom system prompt")

        # Initialize Rain
        rain = RainOrchestrator(
            model_name=args.model,
            max_iterations=args.iterations,
            confidence_threshold=args.confidence,
            system_prompt=system_prompt
        )

        print(f"✅ Rain initialized with model: {args.model}")
        print(f"🎯 Max iterations: {args.iterations}, Confidence threshold: {args.confidence}")

        # Show history if requested
        if args.history:
            history = rain.get_history()
            if history:
                print("\n📚 Reflection History:")
                for i, result in enumerate(history, 1):
                    print(f"{i}. [{result.timestamp.strftime('%H:%M:%S')}] "
                          f"Confidence: {result.confidence:.2f}, "
                          f"Iterations: {result.iteration}")
                    print(f"   {result.content[:100]}...")
            else:
                print("\n📚 No history yet")
            return

        # Interactive mode
        if args.interactive:
            print("\n🌧️ Rain Interactive Mode - Type 'quit' to exit")
            while True:
                try:
                    query = input("\n💬 Ask Rain: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    if not query:
                        continue

                    result = rain.recursive_reflect(query, verbose=args.verbose)
                    print(f"\n🌟 Final Answer (confidence: {result.confidence:.2f}, {result.iteration} iterations, {result.duration_seconds:.1f}s):")
                    print(result.content)

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break

        # Single query mode
        elif args.query:
            result = rain.recursive_reflect(args.query, verbose=args.verbose)
            print(f"\n🌟 Final Answer (confidence: {result.confidence:.2f}, {result.iteration} iterations, {result.duration_seconds:.1f}s):")
            print(result.content)

        else:
            print("\n💡 Use --interactive for chat mode, or provide a query directly")
            print("   Example: python3 rain.py 'What is the capital of France?'")
            print("   Example: python3 rain.py --interactive")
            print("   Example: python3 rain.py --system-file system-prompts/bitcoin-maximalist.txt 'Explain money'")
            print("   Example: python3 rain.py --system-prompt 'You are a helpful coding assistant' 'Debug this Python code'")
            print("\n📝 Check the system-prompts/ folder for example personality profiles!")

    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Rain session ended")


if __name__ == "__main__":
    main()
