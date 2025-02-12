# Intent Classification System

A robust intent classification system built with LangChain and AWS Bedrock, designed to categorize user queries into predefined intent categories using language models.

## Features

- Dynamic intent category configuration
- Pydantic model-based type validation
- LangChain integration for robust language model processing
- AWS Bedrock integration for language model inference
- Comprehensive error handling and logging
- Type hints and documentation for better code maintainability

## Requirements

- Python 3.10+
- AWS account with Bedrock access
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/intent-classification.git
cd intent-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env
```

Configure the following variables in your `.env` file:
```
LLM_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0
MODEL_TEMPERATURE=custom_temperature
MODEL_REGION_NAME=your_region
MODEL_CRED_PROFILE_NAME=your_profile_name
```

## Project Structure

```
intent-classification/
├── identification.py      # Main intent classification implementation
├── llm_model.py          # Bedrock LLM configuration
├── set_logger.py         # Logging configuration
├── requirements.txt      # Project dependencies
├── .env.dev             # Development environment variables
└── .example_env         # Example environment variables template
```

## Usage

### Basic Usage

```python
from llm_model import llm
from identification import IntentClassifier

# Define your intents
intents = [
    "Billing Issue",
    "Technical Support",
    "Account Management",
    "Order Status",
    "General Inquiry"
]

# Sample queries
sample_queries = [
    "I want a refund for my last order",
    "My internet is not working",
    "How do I reset my password?",
    "Where is my package?",
    "What are your working hours?"
]

# Initialize classifier
classifier = IntentClassifier(llm, intents)

# Classify a query
try:
    result = classifier.classify("I want a refund for my last order")
    print(f"Intent: {result['intent']}")
except IntentClassificationError as e:
    print(f"Classification error: {str(e)}")
```

### Custom Configuration

You can customize the intent classification by providing a custom description:

```python
classifier = IntentClassifier(
    model=llm,
    intents=intents,
    default_description="Custom description for intent classification"
)
```

## Error Handling

The system includes comprehensive error handling with custom exceptions:

- `IntentClassificationError`: Raised when classification fails
- `ValueError`: Raised for invalid initialization parameters

## Logging

The system uses Python's logging module with custom configuration. Logs are stored in `intent_classifier.log`.

## AWS Bedrock Configuration

The system uses AWS Bedrock for language model inference. Configure your AWS credentials and Bedrock settings in the `.env` file:

1. Ensure you have AWS Bedrock access
2. Configure AWS credentials by creating an aws-cli profile
3. Select appropriate model ID for your use case
