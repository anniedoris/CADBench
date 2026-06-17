from ._basicprocessor import BasicProcessor
from datasets import Dataset
from typing import Dict, Any, Optional
from PIL import Image
import base64
from io import BytesIO

class VLMProcessor(BasicProcessor):
    def __init__(self, 
                 dataset: Dataset, 
                 resolution: Optional[int] = 512,
                 render_backend: Optional[str] = 'blender',
                 render_style: Optional[str] = 'CAD',
                 system_prompt: Optional[str] = "You are a helpful assistant.",
                 user_prompt: Optional[str] = "Generate the CADQuery code needed to create the CAD for the provided image. Please make sure the final CAD is in a variables called `solid`.",
                 image_type: Optional[str] = 'png',
                 image_field: Optional[str] = None,
                 id_field: str = "file_id"):
        super().__init__(
            dataset=dataset,
            mesh=False,
            point_cloud=False,
            CAD=False,
            image=True,
            image_size=resolution,
            image_field=image_field,
            id_field=id_field,
            render_backend=render_backend,
            render_style=render_style
        )
        
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_type = image_type
    
    @staticmethod
    def encode_image(image: Image.Image, im_type='jpeg') -> str:
        buffered = BytesIO()
        image.save(buffered, format=im_type)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def __call__(self, idx: int) -> Dict[str, Any]:
        input_dict, file_id = super().__call__(idx)
        image = input_dict["image"]
        prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "image_url",
                             "image_url": {"url":
                                 f"data:image/{self.image_type};base64,{self.encode_image(image, im_type=self.image_type)}"}
                             },
                             {"type": "text", "text": self.user_prompt}
                            ]
            }
        ]
        
        return prompt, file_id
