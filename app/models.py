from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import nltk
import re

# Download NLTK sentence tokenizer models if not already done
nltk.download('punkt') # see https://www.nltk.org/api/nltk.tokenize.punkt.html

class EntityExtractor:
    """
    Class for extracting entities from text using a pre-trained NER model.
    """

    def __init__(self, model_name="d4data/biomedical-ner-all"): # see https://huggingface.co/d4data/biomedical-ner-all. A COVID specific model would be better. 
        """
        Initialize the EntityExtractor with a specified model.

        Args:
            model_name (str): Name of the pre-trained model to use.
        """
        # Load the tokenizer and model from the Hugging Face hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name) # see https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
        # Get the list of labels from the model configuration
        self.label_list = self.model.config.id2label
        # Set the maximum length for tokenization
        self.max_length = self.tokenizer.model_max_length

    def get_entities(self, text):
        """
        Get entities from text using the NER model.

        Args:
            text (str): Input text to extract entities from.

        Returns:
            list: List of entities extracted from the text, each with its corresponding label and confidence.
        """
        # Tokenize the input text and prepare it for the model (returns in pytorch tensor format)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Apply softmax to get probabilities for each token
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get the predicted labels and their probabilities
        predictions = torch.argmax(probs, dim=2)
        confidences = torch.max(probs, dim=2).values

        # Convert token IDs to token strings
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = []
        current_entity = []
        current_label = None
        current_confidences = []
        # When merging subword tokens into complete entities, we compute the average confidence of the merged entity. 
        # The rationale for using the average confidence is to provide a single confidence score that reflects the model's 
        # certainty about the entire entity, considering all subword tokens that make up the entity.
        for token, prediction, confidence in zip(tokens, predictions[0], confidences[0]):
            label = self.label_list[prediction.item()]
            conf = confidence.item()
            # Debugging: log each token, label, and confidence
            # print(f"Token: {token}, Label: {label}, Confidence: {conf}")
            if label != "O":  # Ignore the non-entity label
                if token.startswith("##"):
                    # Append the sub-token to the current entity
                    current_entity.append(token[2:])
                    current_confidences.append(conf)
                else:
                    if current_entity:
                        # Calculate average confidence for the current entity
                        avg_confidence = sum(current_confidences) / len(current_confidences)
                        entities.append(("".join(current_entity), current_label, avg_confidence))
                    # Start a new entity
                    current_entity = [token]
                    current_label = label
                    current_confidences = [conf]
            else:
                if current_entity:
                    # End of an entity, add it to the list
                    avg_confidence = sum(current_confidences) / len(current_confidences)
                    entities.append(("".join(current_entity), current_label, avg_confidence))
                    current_entity = []
                    current_label = None
                    current_confidences = []
        if current_entity:
            # Add the last entity if any
            avg_confidence = sum(current_confidences) / len(current_confidences)
            entities.append(("".join(current_entity), current_label, avg_confidence))

        return self.merge_entities(entities)

    def merge_entities(self, entities):
        """
        Merge adjacent entities that should logically form a single entity.

        Args:
            entities (list): List of extracted entities with their labels and confidences.

        Returns:
            list: List of merged entities.
        """
        if not entities:
            return entities

        merged_entities = []
        current_entity, current_label, current_confidence = entities[0]
        for entity, label, confidence in entities[1:]:
            # Merge if the label is the same and the entities are adjacent (connected by hyphens or spaces)
            if label == current_label and (entity.startswith('-') or entity.startswith(' ') or current_entity.endswith('-') or current_entity.endswith(' ')):
                current_entity += entity
                # Update the confidence by averaging
                current_confidence = (current_confidence + confidence) / 2
            else:
                merged_entities.append((current_entity, current_label, current_confidence))
                current_entity, current_label, current_confidence = entity, label, confidence

        merged_entities.append((current_entity, current_label, current_confidence))
        return merged_entities

    def get_entities_from_long_text(self, text):
        """
        Process long text by splitting it into smaller chunks and extracting entities from each chunk.

        Args:
            text (str): Input long text to extract entities from.

        Returns:
            list: List of entities extracted from the text.
        """
        entities = []
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            # Extract entities from each sentence
            entities.extend(self.get_entities(sentence))
        return entities

    def extract_abstract(self, text):
        """
        Extract the abstract section from the text.

        Args:
            text (str): Full text extracted from the PDF.

        Returns:
            str: Abstract section of the text.
        """
        # Use regular expressions to find the abstract section
        match = re.search(r'(?i)abstract[\s\:\u2002-\u200b]*', text)
        if match:
            start_index = match.end()
            end_index = start_index + 5000  # Adjust length as needed
            abstract = text[start_index:end_index].strip()
            # Print the first 500 characters of the extracted abstract for debugging
            print(f"Extracted abstract: {abstract[:500]}...")
            return abstract
        else:
            # Fall back to extracting the first N sentences if no abstract section is found
            sentences = nltk.sent_tokenize(text)
            abstract = " ".join(sentences[:20])  # N = 20. Adjust the number of sentences as needed
            # Print the fallback abstract for debugging
            print(f"Extracted abstract (fallback): {abstract}")
            return abstract

    def extract_context(self, text, entities, context_window=50):
        """
        Extract context around each identified entity in the text.

        Args:
            text (str): Original text.
            entities (list): List of entities extracted from the text.
            context_window (int): Number of tokens to include before and after the entity.

        Returns:
            list: List of entities with their context.
        """
        context_info = []
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        unique_entities = set()

        for entity, label, confidence in entities:
            entity_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity))
            start_index = None
            for idx in range(len(token_ids) - len(entity_token_ids) + 1):
                if token_ids[idx:idx + len(entity_token_ids)] == entity_token_ids:
                    start_index = idx
                    break

            if start_index is None:
                continue

            end_index = start_index + len(entity_token_ids)

            # Extract context tokens
            context_start = max(0, start_index - context_window)
            context_end = min(len(tokens), end_index + context_window)
            context_tokens = tokens[context_start:context_end]
            context_text = self.tokenizer.convert_tokens_to_string(context_tokens)

            entity_info = {
                "entity": entity,
                "context": context_text,
                "model_label": label,
                "confidence": confidence,
                "start": start_index,
                "end": end_index
            }
            # Check for duplicates
            entity_key = (entity, start_index, end_index)
            if entity_key not in unique_entities:
                unique_entities.add(entity_key)
                context_info.append(entity_info)

        return context_info
