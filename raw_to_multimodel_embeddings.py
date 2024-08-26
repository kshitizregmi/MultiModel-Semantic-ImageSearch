import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from vertexai.vision_models import Image, MultiModalEmbeddingModel

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with necessary columns and cleaned data.
    """
    df = pd.read_csv(
        file_path, 
        usecols=['p_id', 'name', 'price', 'colour', 'brand', 'img', 'description'],
        na_values=['nan', 'na', np.nan]
    )
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.p_id = df.p_id.astype(int)
    return df

def remove_html_tags(text):
    """
    Remove HTML tags from a given text.
    
    Args:
        text (str): Text containing HTML tags.
        
    Returns:
        str: Cleaned text without HTML tags.
    """
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def clean_descriptions(df):
    """
    Apply HTML tag removal to the 'description' column of the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'description' column containing HTML content.
        
    Returns:
        pd.DataFrame: DataFrame with cleaned 'description' column.
    """
    df['description'] = df['description'].apply(remove_html_tags)
    return df

def sample_data(df, n=5, random_state=0):
    """
    Sample a subset of the DataFrame.
    
    Args:
        df (pd.DataFrame): The original DataFrame.
        n (int): Number of samples to draw.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    dfs = df.sample(n, random_state=random_state)
    dfs.dropna(inplace=True)
    return dfs

def get_image_text_embedding(model, sliced):
    """
    Get image and text embeddings using a pre-trained model.
    
    Args:
        model: The embedding model to use.
        sliced (pd.Series): A row of the DataFrame with image URL and text data.
        
    Returns:
        dict: Metadata with image and text embeddings added.
    """
    print(sliced['img'])



    
    response = requests.get(sliced['img'])
    image_bytes = response.content
    image = Image(image_bytes=image_bytes)
    
    contextual_text = (
        sliced['name'] + " " + 
        sliced['colour'] + " " + 
        sliced['brand'] + " " + 
        sliced['description']
    )
    
    embeddings = model.get_embeddings(
        image=image,
        contextual_text=contextual_text,
        dimension=128
    )
    
    metadata = sliced.to_dict()
    metadata.update({
        "image_embedding": np.array(embeddings.image_embedding),
        "text_embedding": np.array(embeddings.text_embedding)
    })
    
    return metadata

def compute_embeddings(df, model):
    """
    Compute image and text embeddings for each row in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        model: The embedding model to use.
        
    Returns:
        pd.DataFrame: DataFrame with embeddings and combined embeddings.
    """
    embedding_data = [get_image_text_embedding(model, df.iloc[i]) for i in range(len(df))]
    emb = pd.DataFrame(embedding_data)
    emb['combined_embeddings'] = (emb['image_embedding'] + emb['text_embedding']) / 2
    return emb

if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data("data/Fashion Dataset.csv")
    df = clean_descriptions(df)
    
    # Sample data and save to CSV
    dfs = sample_data(df)
    dfs.to_csv("data/myntra-small.csv", index=False)
    
    # Load sampled data
    df = pd.read_csv("data/myntra-small.csv")
    df.dropna(inplace=True)
    
    # Initialize the embedding model
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    
    # Compute and save embeddings
    emb = compute_embeddings(df, model)
    emb['combined_embeddings'] = emb['combined_embeddings'].apply(list)
    emb['text_embedding'] = emb['text_embedding'].apply(list)
    emb['image_embedding'] = emb['image_embedding'].apply(list)
    
    emb.to_csv('data/embeddings.csv', index=False)
