import base64
import requests
import numpy as np
from io import BytesIO
from qdrant_client import QdrantClient
from PIL import Image as PILImage
from IPython.display import display, HTML
from vertexai.vision_models import Image, MultiModalEmbeddingModel

class EmbeddingInference:
    """
    Class responsible for performing search and inference operations on embeddings.
    """
    def __init__(self, db_path: str = 'vectordb'):
        """
        Initialize the EmbeddingInference.

        :param db_path: Path to the Qdrant vector database directory.
        """
        self.client = QdrantClient(path=db_path)
        self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

    def get_image_text_embedding(self, query: str = None, image: Image = None) -> np.ndarray:
        """
        Generate embeddings for the provided query and/or image.

        :param query: Text query for which to generate embeddings.
        :param image: Image for which to generate embeddings.
        :return: Generated embedding as a numpy array.
        """
        embeddings = self.model.get_embeddings(image=image, contextual_text=query, dimension=128)
        if query and not image:
            return embeddings.text_embedding
        elif image and not query:
            return embeddings.image_embedding
        else:
            return (np.array(embeddings.image_embedding) + np.array(embeddings.text_embedding)) / 2

    def search(self, collection_name: str, query_vector: np.ndarray, limit: int = 3):
        """
        Search the specified collection for the closest embeddings.

        :param collection_name: Name of the collection to search.
        :param query_vector: Query vector to search for.
        :param limit: Number of closest points to return.
        :return: List of closest points.
        """
        return self.client.search(collection_name=collection_name, query_vector=query_vector, limit=limit)

    @staticmethod
    def fetch_images(scored_points):
        """
        Fetch images from the URLs present in the scored points' payloads.

        :param scored_points: List of scored points containing image URLs in their payloads.
        :return: List of fetched images.
        """
        images = []
        for point in scored_points:
            img_url = point.payload.get('img')
            if img_url:
                try:
                    response = requests.get(img_url)
                    image = PILImage.open(BytesIO(response.content))
                    images.append(image)
                except Exception as e:
                    print(f"Error processing image {img_url}: {e}")
        return images

    @staticmethod
    def display_images_horizontally(images):
        """
        Display images horizontally in a Jupyter notebook.

        :param images: List of images to display.
        """
        html = '<div style="display: flex;">'
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            html += f'<img src="data:image/png;base64,{img_str}" style="height: 400px; width: 300px; margin-right: 10px;">'
        html += '</div>'
        display(HTML(html))




# Initialize the inference engine
inference = EmbeddingInference()

# Text to Image Search
query = "White Embroidered Kurta"
query_vector = inference.get_image_text_embedding(query=query)
hits = inference.search(collection_name="text_embedding", query_vector=query_vector, limit=3)
images = inference.fetch_images(hits)
inference.display_images_horizontally(images)

# # Image to Image Search
# image_url = 'https://example.com/sample_image.jpeg'
# response = requests.get(image_url)
# image_bytes = response.content
# image = Image(image_bytes=image_bytes)
# query_vector = inference.get_image_text_embedding(image=image)
# hits = inference.search(collection_name="image_embedding", query_vector=query_vector, limit=3)
# images = inference.fetch_images(hits)
# inference.display_images_horizontally(images)

# # Text + Image to Image Search
# image_url = "https://example.com/sample_lehenga.jpg"
# response = requests.get(image_url)
# image_bytes = response.content
# image = Image(image_bytes=image_bytes)
# query_vector = inference.get_image_text_embedding(query="Green Lehenga Choli", image=image)
# hits = inference.search(collection_name="combined_embeddings", query_vector=query_vector, limit=3)
# images = inference.fetch_images(hits)
# inference.display_images_horizontally(images)
