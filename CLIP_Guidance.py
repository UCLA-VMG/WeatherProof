import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
    
class CLIP_Guidance(nn.Module):
    """
    This class is added to the backbone in MMSeg to extract the CLIP-Informed Prior.
    """

    def __init__(self, model_type="ViT-B/32"):
        super().__init__()
        self.weight_proj = nn.Linear(512, 13)
        self.clip_model, _ = clip.load(model_type)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        input_text = ["rainy weather, rain", "snowy weather, snow", "foggy weather, fog", 
                      "clear weather, clear", "sunny weather, sun", "cloudy weather, cloud", 
                      "overcast weather, overcast clouds", "partly cloudy weather, partly cloudy", 
                      "misty weather, mist", "hazy weather, haze", "downpour weather, downpour rain", 
                      "blizzard weather, blizzard", "precipitation weather, precipitation"]
        with torch.no_grad():
            text = clip.tokenize(input_text).to('cuda')
            text_features = self.clip_model.encode_text(text).float()
        self.weathered_features = text_features.unsqueeze(0)

    def extract_clip(self, adverse_image):
        """
        The function lies within a CLIP guidance class. This function extracts the CLIP encoding for the input image(s).

        Inputs:
            input_image: Input image(s) for which the CLIP encoding is to be extracted
        Output:
            features: Output CLIP guidance features of the input image(s)
        """
        # Step 1: Compute CLIP image encodings
        with torch.no_grad():
            adverse_image_features = self.clip_model.encode_image(adverse_image)
            # Step 2: Ensure correct dimensions
            adverse_image_features = adverse_image_features.unsqueeze(1)

        # Step 2: Ensure correct type
        if adverse_image.dtype != torch.float:
            adverse_image = adverse_image.float()
        if adverse_image_features.dtype != torch.float:
            adverse_image_features = adverse_image_features.float()

        B = adverse_image.shape[0]
        # Step 3: Find weights associated with text features for each image encoding using a projection layer
        adverse_weights = (self.weight_proj(adverse_image_features))
        adverse_weights = adverse_weights.squeeze(1).unsqueeze(2)
        adverse_weights = adverse_weights.repeat(1, 1, 512)
        weathered_features = self.weathered_features.repeat(B, 1, 1)
        # Step 4 (Adverse Image): Multiply weights to text features and sum them
        weathered_features = weathered_features * adverse_weights
        weathered_features = torch.sum(weathered_features, dim=1, keepdim=True)
        # Step 5: Concatenate the image encodings with the summed text features
        features = torch.cat([adverse_image_features, weathered_features], dim=2)
        # Step 6: Return the CLIP-Informed Prior
        return features
    
