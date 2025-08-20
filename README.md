# BRIA Attribution Model 

This repository contains a Jupyter notebook demonstrating how to use the BRIA attribution model for image embedding generation and attribution API calls.

## Overview

The `code_example.ipynb` notebook provides a complete workflow for:
1. Setting up and running the BRIA attribution model using Docker and Triton inference server
2. Generating image embeddings using the BRIA model
3. Sending attribution API requests with the generated embeddings

## Prerequisites

Before running this example, ensure you have the following installed:

- **Docker**
- **Python 3** with the packages (pip install -r ./requirements.txt --use-deprecated=legacy-resolver)
- **NVIDIA GPU** with CUDA support (recommended)

## Setup Instructions

### 1. AWS Credentials Configuration

You'll need to configure your AWS credentials to access the BRIA model from Amazon ECR. Replace the placeholder values in the notebook:

```python
os.environ['AWS_ACCESS_KEY_ID'] = "<AWS_ACCESS_KEY_ID>"
os.environ['AWS_SECRET_ACCESS_KEY'] = "<AWS_SECRET_ACCESS_KEY>"
```

**Required substitutions:**
- `<AWS_ACCESS_KEY_ID>`: Your AWS access key ID
- `<AWS_SECRET_ACCESS_KEY>`: Your AWS secret access key

### 2. Model Deployment

The notebook will automatically:
- Login to Amazon ECR
- Pull the BRIA attribution model Docker image
- Start the Triton inference server with the model

The model will be available on:
- Port 8000: HTTP endpoint
- Port 8001: Metrics endpoint  
- Port 8002: GRPC endpoint

### 3. Client Configuration

#### Required Substitutions in the Notebook:

1. **BRIA Embeddings URL:**
   ```python
   url = '<BRIA_EMBEDDINGS_URL>:8000'
   ```
   Replace `<BRIA_EMBEDDINGS_URL>` with your server's IP address or hostname (localhost - in case of the same machine).

2. **Image Path:**
   ```python
   image_path = '<IMAGE_PATH>'
   ```
   Replace `<IMAGE_PATH>` with the path to your input image file.

3. **API Token:**
   ```python
   "agent": '<API-TOKEN>',
   "api_token": '<API-TOKEN>'
   ```
   Replace `<API-TOKEN>` with your BRA API token.

## Usage

### Step 1: Run the Docker Setup Cell

Execute the first cell to:
- Configure AWS credentials
- Login to ECR
- Pull and run the BRIA attribution model container

### Step 2: Install Dependencies

Run the installation cell to import required libraries:
- `requests` for API calls
- `tritonclient.http` for Triton inference
- `PIL` for image processing
- `embedder` module (BRIAEmbedder class)

### Step 3: Generate Embeddings

Execute the inference request cell to:
- Connect to the Triton server
- Load and process your image
- Generate embeddings using the BRIA model

### Step 4: Send Attribution Request

Run the attribution API cell to:
- Create a unique embedding ID
- Send the embeddings to the attribution service
- Handle the API response

## Configuration Options

### Model Parameters

The notebook uses these default parameters:
- **Model Version**: 2.3
- **Model Name**: 'replace-backgroud'
- **Agent**: Your API token

## API Request Format

The attribution API expects the following JSON structure:

```json
{
  "embeddings_base64": "<base64_encoded_embeddings>",
  "embeddings_uid": "<unique_identifier>",
  "model_version": 2.3,
  "agent": "<API-TOKEN>",
  "model_name": "replace-backgroud"
}
```
