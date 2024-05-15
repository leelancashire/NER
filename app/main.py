import os
import json
from flask import Blueprint, request, jsonify, current_app
from .utils import extract_text_from_pdf, save_file
from .models import EntityExtractor

# Create a new Blueprint named 'main'. This will hold our routes and views for this module.
bp = Blueprint('main', __name__)  # see https://flask.palletsprojects.com/en/1.1.x/tutorial/views/
# Initialize the entity extractor, which will be used to process incoming text data.
extractor = EntityExtractor()

# Define a route for the Blueprint. This route will handle POST requests to '/api/v1/extract'.
@bp.route('/api/v1/extract', methods=['POST'])
def extract_entities():
    """
    Extract medical entities from a PDF document.

    Returns:
        Response: JSON response containing the extracted entities and their context.
    """
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']

    if file.filename == '':
        return "Empty filename", 400

    if not file.filename.endswith('.pdf'):
        return "Unsupported file type", 415

    # Save the PDF file
    file_path = save_file(file, current_app.config['UPLOAD_FOLDER'])

    # Extract text from PDF
    text = extract_text_from_pdf(file_path)
    # print(f"Extracted text: {text[:500]}...")  # Print the first 500 characters for debugging

    # Extract the abstract section from the text
    abstract = extractor.extract_abstract(text)
    # print(f"Extracted abstract: {abstract}")

    # Perform NER on extracted abstract
    entities = extractor.get_entities_from_long_text(abstract)
    # print(f"Extracted entities: {entities}")

    context_info = extractor.extract_context(abstract, entities)
    # print(f"Context info: {context_info}")

    # Save the JSON response to a file
    output_file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'extracted_entities.json')
    try:
        with open(output_file_path, 'w') as f:
            json.dump(context_info, f, indent=4)
        print(f"Successfully wrote extracted entities to {output_file_path}")
    except Exception as e:
        print(f"Failed to write extracted entities to file: {e}")

    return jsonify(context_info), 200


