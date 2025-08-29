import traceback
import random
import os
from dotenv import load_dotenv
from json.decoder import JSONDecodeError
from typing import Dict, Any
from pydantic import BaseModel

from results_utils import ResultsAgent, save_results

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langsmith import Client, evaluate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

DATASET_NAME = "DVD Documents"
MODEL = "gemini-2.5-pro"
SPLITS = ["base"]
NUM_RUNS = 1

GOOGLE_SHEETS_CREDENTIALS_PATH = "credentials.json"
SPREADSHEET_ID = "1T6Rvi1X61Tj_L3VMt2TNbUUbGtZHzgf3MA5aW9ilYAM"
PROJECT_NAME = f"{DATASET_NAME} - {MODEL} - {random.randint(0, 1000)}"

class CategoryOutput(BaseModel):
    category: str

def get_prompt():
    """Load the Simple prompt for universal extraction"""
    try:
        with open(f"Simple.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading Simple prompt file: {e}")

def map_input_to_messages_lambda(inputs: dict):
    """Convert inputs to LangChain messages format"""
    image_data = inputs.get("image", {})
    prompt_text = get_prompt()
    
    # Create multimodal content with text and image
    multimodal_content = [
        {"type": "text", "text": prompt_text}
    ]
    
    # Add image if present
    if image_data:
        multimodal_content.append(image_data)
    
    messages = [
        HumanMessage(content=multimodal_content),
    ]
    
    return messages

def aggregate_results_with_certainty(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    
    # Get all field names from the first result
    field_names = list(results[0].keys())
    aggregated = {}
    
    for field_name in field_names:
        # Collect all values for this field
        field_values = []
        for result in results:
            value = result.get(field_name, "")
            field_values.append(str(value))
        
        # Count occurrences
        from collections import Counter
        value_counts = Counter(field_values)
        
        # Get most frequent value
        most_frequent_value = value_counts.most_common(1)[0][0] if value_counts else ""
        
        # Calculate certainty (all runs agree)
        certainty = len(value_counts) == 1 and len(results) > 1
        
        # Store result with certainty information
        aggregated[field_name] = {
            'value': most_frequent_value,
            'certainty': certainty
        }
    
    return aggregated

def multiple_runs_extraction(llm_chain, formatted_inputs: dict, num_runs: int = NUM_RUNS) -> Dict[str, Any]:
    results = []
    
    for i in range(num_runs):
        try:
            result = llm_chain.invoke(formatted_inputs)
            results.append(result)
        except Exception as e:
            print(f"Error in run {i+1}: {e}")
            # Continue with other runs even if one fails
            continue
    
    if not results:
        raise ValueError("All extraction runs failed")
    
    # Aggregate results with certainty
    aggregated_result = aggregate_results_with_certainty(results)
    
    # Return the aggregated results directly
    return aggregated_result

def simple_target_function(llm_chain):
    def target(inputs: dict) -> dict:
        if "image" not in inputs:
            inputs = inputs["inputs"]
        if "image" not in inputs:
            raise ValueError("Missing 'multimodal_prompt' in inputs")
        
        formatted_inputs = {"image": inputs["image"]}
        
        # Run extraction once - LangSmith will handle repetitions
        result = llm_chain.invoke(formatted_inputs)

        if hasattr(result, 'model_dump'):
            # It's a Pydantic model
            postprocessed_results = result.model_dump()
        elif isinstance(result, dict):
            # It's already a dict
            postprocessed_results = result
        else:
            # Fallback - try to convert to dict
            try:
                postprocessed_results = dict(result)
            except Exception as e:
                print(f"Failed to convert result to dict: {e}")
                # Return empty dict as fallback
                postprocessed_results = {}
        
        # Convert to dictionary format expected by evaluators
        results_dict = {}
        for field_name, field_value in postprocessed_results.items():
            if field_value is None:
                # Handle None values - convert to empty string
                results_dict[field_name] = ""
            else:
                str_value = str(field_value) if field_value is not None else ""
                # Handle "nan", "NaN", "NAN" strings specifically
                if str_value.lower() in ['nan', 'none', 'null', 'n/a', 'na']:
                    results_dict[field_name] = ""
                else:
                    results_dict[field_name] = str_value
        
        return results_dict
    
    return target

def field_match(outputs: dict, reference_outputs: dict) -> dict:
    """Simple field matching for classification task"""
    try:
        if not outputs or not isinstance(outputs, dict):
            return {"score": 0.0, "key": "field_match"}
        
        predicted_category = outputs.get("category", "")
        reference_category = reference_outputs.get("category", "")
        
        # Check if prediction matches reference
        is_correct = predicted_category == reference_category
        
        return {
            "score": 1.0 if is_correct else 0.0,
            "key": "field_match"
        }
    except Exception as e:
        print(f"Error in field_match: {e}")
        return {"score": 0.0, "key": "field_match"}

def main():
    client = Client(api_key=LANGSMITH_API_KEY)

    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.4,
        max_tokens=4096,
        max_output_tokens=1024,
        thinking_budget=2048 if "2.5" in MODEL else None,
    )

    runnable = RunnableLambda(map_input_to_messages_lambda)
    llm_retry = llm.with_retry(retry_if_exception_type=(Exception, JSONDecodeError), stop_after_attempt=5)
    json_parser = JsonOutputParser(pydantic_object=CategoryOutput)

    def llm_chain_factory():
        return runnable | llm_retry | json_parser

    print(f"\nStarting run on dataset '{DATASET_NAME}' with project name '{PROJECT_NAME}'...")
    print(f"Using LangSmith's num_repetitions={NUM_RUNS} for certainty calculation...")

    target = simple_target_function(llm_chain_factory())
    
    try:
        results = evaluate(
            target,
            data=client.list_examples(dataset_name=DATASET_NAME, splits=SPLITS),
            evaluators=[field_match],
            experiment_prefix=f"{MODEL} ",
            client=client,
            max_concurrency=20,
            num_repetitions=NUM_RUNS,
        )

        print("\nRun on dataset completed successfully!")
        results_path = f"results/{PROJECT_NAME}_results.csv"
        save_results(results, results_path)
        results_agent = ResultsAgent(
            spreadsheet_id=SPREADSHEET_ID,
            credentials_path=GOOGLE_SHEETS_CREDENTIALS_PATH,
        )
        results_agent.upload_results(results_path)

    except Exception as e:
        print(f"\nAn error occurred during the run on dataset")
        traceback.print_exc()
        print("\nRun on dataset failed.")

if __name__ == "__main__":
    main() 