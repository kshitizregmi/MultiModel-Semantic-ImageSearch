import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class EmbeddingIndexer:
    """
    Class responsible for creating collections and indexing embeddings into Qdrant.
    """
    def __init__(self, csv_path: str, db_path: str = 'vectordb'):
        """
        Initialize the EmbeddingIndexer.

        :param csv_path: Path to the CSV file containing the embeddings.
        :param db_path: Path to the Qdrant vector database directory.
        """
        self.csv_path = csv_path
        self.db_path = db_path
        self.client = QdrantClient(path=db_path)
        self.df = None

    def load_and_prepare_data(self):
        """
        Load and prepare data from the CSV file.
        """
        self.df = pd.read_csv(self.csv_path)
        self.df.drop_duplicates(inplace=True)
        self.df['uuid'] = self.df.apply(lambda _: str(uuid.uuid4()), axis=1)
        self.df['image_embedding'] = self.df['image_embedding'].apply(eval)
        self.df['text_embedding'] = self.df['text_embedding'].apply(eval)
        self.df['combined_embeddings'] = self.df['combined_embeddings'].apply(eval)

    def create_collections(self):
        """
        Create collections in Qdrant for different embedding types.
        """
        collections = ["text_embedding", "image_embedding", "combined_embeddings"]
        for collection in collections:
            try:
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=128, distance=Distance.COSINE)
                )
            except Exception as e:
                print(f"Error creating collection {collection}: {e}")

    def prepare_points(self):
        """
        Prepare data points for insertion into Qdrant.
        """
        metadata = self.df.iloc[:, :7].to_dict('records')
        ids = self.df['uuid'].tolist()
        text_embeddings = self.df['text_embedding'].tolist()
        image_embeddings = self.df['image_embedding'].tolist()
        combined_embeddings = self.df['combined_embeddings'].tolist()

        self.text_points = [
            PointStruct(id=uid, vector=emb, payload=met)
            for uid, emb, met in zip(ids, text_embeddings, metadata)
        ]
        self.image_points = [
            PointStruct(id=uid, vector=emb, payload=met)
            for uid, emb, met in zip(ids, image_embeddings, metadata)
        ]
        self.combined_embeddings_points = [
            PointStruct(id=uid, vector=emb, payload=met)
            for uid, emb, met in zip(ids, combined_embeddings, metadata)
        ]

    def index_data(self):
        """
        Insert prepared data points into the respective Qdrant collections.
        """
        self.client.upsert(collection_name="text_embedding", points=self.text_points)
        self.client.upsert(collection_name="image_embedding", points=self.image_points)
        self.client.upsert(collection_name="combined_embeddings", points=self.combined_embeddings_points)



indexer = EmbeddingIndexer(csv_path='data/embeddings.csv')
indexer.load_and_prepare_data()
indexer.create_collections()
indexer.prepare_points()
indexer.index_data()
