import os
import pytest
from app import create_app

@pytest.fixture
def client():
    """
    Flask test client fixture.
    """
    app = create_app()
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = '/tmp'
    with app.test_client() as client:
        yield client

def test_extract_entities(client):
    """
    Test the entity extraction endpoint with a sample PDF.
    """
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'Enfothelial dysfunction.pdf')
    assert os.path.exists(pdf_path), f"Test PDF file does not exist at {pdf_path}"

    with open(pdf_path, 'rb') as f:
        response = client.post('/api/v1/extract', data={'file': f})
    
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    for entity in data:
        assert 'entity' in entity
        assert 'context' in entity
        assert 'start' in entity
        assert 'model_label' in entity
        assert 'confidence' in entity
        assert 'end' in entity
