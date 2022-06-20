import torch

ImageBatch = torch.Tensor     # Batch of RGB or grayscale images, shape = (batch_size, num_channels, height, width), dtype = torch.float32
Logits = torch.Tensor         # Image segmentation logit, shape = (batch_size, num_classes, height, width), dtype = torch.float32
SemanticBatch = torch.Tensor  # Semantic segmentation mask, shape = (batch_size, height, width), dtype = torch.long