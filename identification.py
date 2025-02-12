from typing import List, Literal, Optional, TypeVar
from pydantic import BaseModel, Field, create_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from set_logger import logger

# Type variables
ModelType = TypeVar('ModelType', bound=BaseModel)

class IntentClassificationError(Exception):
    """Custom exception for intent classification errors."""
    pass

class IntentClassifier:
    """
    A class for classifying user queries into predefined intent categories.
    
    This class uses LangChain and Language Models to process user queries and
    determine their intent based on a predefined set of possible intents.
    
    Attributes:
        model (BaseLanguageModel): The language model to use for classification
        intents (List[str]): List of possible intent categories
        DEFAULT_DESCRIPTION (str): Default description for the intent field
    """
    
    def __init__(
        self,
        model: BaseLanguageModel,
        intents: List[str],
        default_description: str = "Identify the intent of the user query"
    ) -> None:
        """
        Initialize the IntentClassifier.

        Args:
            model: Language model to use for classification
            intents: List of possible intent categories
            default_description: Default description for the intent field
        
        Raises:
            ValueError: If intents list is empty or model is None
        """
        if not intents:
            raise ValueError("Intents list cannot be empty")
        if model is None:
            raise ValueError("Model cannot be None")
            
        self.model = model
        self.intents = intents
        self.DEFAULT_DESCRIPTION = default_description
        logger.info(f"Initialized IntentClassifier with {len(intents)} intents")

    def create_intent_model(
        self,
        description: Optional[str] = None
    ) -> ModelType:
        """
        Create a Pydantic model for intent classification.

        Args:
            description: Optional custom description for the intent field

        Returns:
            A Pydantic model with an intent field
        """
        try:
            # Create Literal type from intents
            intent_literal = Literal[tuple(self.intents)]  # type: ignore
            
            # Set field description
            field_description = description or self.DEFAULT_DESCRIPTION
            field_definition = (
                intent_literal,
                Field(description=field_description)
            )
            
            # Create and return the model
            return create_model(
                "IntentClassification",
                intent=field_definition,
                __base__=BaseModel
            )
        except Exception as e:
            logger.error(f"Error creating intent model: {str(e)}")
            raise IntentClassificationError(f"Failed to create intent model: {str(e)}")

    def _create_parser(self, model_config: ModelType) -> JsonOutputParser:
        """
        Create a JSON parser for the intent model.

        Args:
            model_config: Pydantic model configuration

        Returns:
            Configured JsonOutputParser
        """
        return JsonOutputParser(pydantic_object=model_config)

    def _create_prompt(self, parser: JsonOutputParser) -> PromptTemplate:
        """
        Create a prompt template for intent classification.

        Args:
            parser: Configured JSON parser

        Returns:
            PromptTemplate object
        """
        return PromptTemplate(
            template="Analyze the following user query and determine its intent.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

    def _create_chain(self, prompt: PromptTemplate, parser: JsonOutputParser):
        """
        Create the classification chain.

        Args:
            prompt: Configured prompt template
            parser: Configured JSON parser

        Returns:
            LangChain chain
        """
        return prompt | self.model | parser

    def classify(self, query: str) -> ModelType:
        """
        Classify a user query into one of the predefined intent categories.

        Args:
            query: User query to classify

        Returns:
            Classification result containing the identified intent

        Raises:
            IntentClassificationError: If classification fails
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Create classification chain
            model_config = self.create_intent_model()
            parser = self._create_parser(model_config)
            prompt = self._create_prompt(parser)
            chain = self._create_chain(prompt, parser)
            
            # Process query
            result = chain.invoke(query)
            logger.info(f"Classification result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise IntentClassificationError(f"Failed to classify query: {str(e)}")

def main():
    """Example usage of the IntentClassifier."""
    from llm_model import llm  # Import your LLM model
    
    # Define sample intents
    intents = [
        "Billing Issue",
        "Technical Support",
        "Account Management",
        "Order Status",
        "General Inquiry"
    ]
    
    # Create classifier
    classifier = IntentClassifier(llm, intents)
    
    # Sample queries
    sample_queries = [
        "I want a refund for my last order",
        # "My internet is not working",
        # "How do I reset my password?",
        # "Where is my package?",
        # "What are your working hours?"
    ]
    
    # Process queries
    for query in sample_queries:
        try:
            result = classifier.classify(query)
            print(f"\nQuery: {query}")
            print(f"Intent: {result['intent']}")
        except IntentClassificationError as e:
            print(f"Error processing query '{query}': {str(e)}")

if __name__ == "__main__":
    main()