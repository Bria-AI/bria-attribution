import os
import time
from enum import Enum
from typing import Union
from PIL import Image
import numpy as np
import tritonclient.http as httpclient
from PIL.Image import Image as ImageType
from transformers.models.clip.feature_extraction_clip import CLIPFeatureExtractor
from transformers.models.clip.tokenization_clip import CLIPTokenizer



BRIA_ATTRIBUTION_CLIENT_DIR = "bria_attribution_model_client"
models_base_path = os.path.dirname(os.path.realpath(__file__))

class AttributionModel(Enum):
    bria = "bria_attribution_model"

class BRIAEmbedder:
    def __init__(self, triton_client):
        self.clip_feature_extractor = {}
        self.tokenizer = {}
        self.triton_client = triton_client
        self._load_model()

    def _load_model(self) -> None:
        """Load models into memory
        Returns:
            None
        """
        self.clip_feature_extractor[AttributionModel.bria.value] = CLIPFeatureExtractor.from_pretrained(
            f"{models_base_path}/{BRIA_ATTRIBUTION_CLIENT_DIR}/feature_extractor"
        )
        self.tokenizer[AttributionModel.bria.value] = CLIPTokenizer.from_pretrained(
            f"{models_base_path}/{BRIA_ATTRIBUTION_CLIENT_DIR}/tokenizer"
            )

    def run_on_image(
            self,
            image: ImageType,
            model=AttributionModel.bria.value,
            normalize: bool = False,
    ):
        """Run inference on a image
        Args:
            normalize:
            model:
            image (ImageType): pil image to be embedded
        Returns:
            np.array: outputs an embedding array
        """
        if len(image.mode) != 3:
            image = image.convert("RGB")
        inputs_images = self.clip_feature_extractor[model](
            images=image, return_tensors="np", padding=True
        )

        image_embeds = self.sagemaker_inference(
            [inputs_images["pixel_values"]], model
        )
        return [image_embed.decode() for image_embed in image_embeds]
        # if normalize:
        #     image_embeds = self.norm_embedding(image_embeds)

        # return b64encode(image_embeds).decode()

    def sagemaker_inference(
            self, inputs, model_name, model_version="1", dtype="FP32"
    ) -> Union[np.ndarray, None]:
        inputs_ = []
        for i, image_pixels in enumerate(inputs):
            input0 = httpclient.InferInput(
                f"input__{i}", tuple(image_pixels.shape), dtype
            )
            input0.set_data_from_numpy(image_pixels)
            inputs_.append(input0)

        output_name = "output__0"
        outputs = [httpclient.InferRequestedOutput(name=output_name)]

        (
            request_body,
            header_length,
        ) = httpclient.InferenceServerClient.generate_request_body(
            inputs_, outputs=outputs
        )
        # health_ctx = triton_client.is_server_ready(headers=headers)
        # print("Is server ready - {}".format(health_ctx))

        # status_ctx = triton_client.is_model_ready(model_name, "1", headers)
        # print("Is model ready - {}".format(status_ctx))

        t = time.time()
        print(f"Sending request to {model_name}...")
        response = self.triton_client.infer(model_name, inputs_, outputs=outputs)

        print({f"{model_name}_infer": time.time() - t})
        return response.as_numpy(output_name)

    @staticmethod
    def norm_embedding(embedding):
        return embedding / embedding.norm(dim=-1, keepdim=True)


if __name__ == "__main__":
    url = 'localhost:8000'
    triton_client = httpclient.InferenceServerClient(
    url=url,
    # ssl=True,
    # ssl_context_factory=gevent.ssl._create_default_https_context,
    )
    image = Image.open(
        "/mnt/models/nvcf-asset-manager-example/increase_resolution/workspace/data/images/MicrosoftTeams-image.png"
    )
    bria_pipeline = BRIAEmbedder(triton_client)
    t = bria_pipeline.run_on_image(image)
    print(t)
