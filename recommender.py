import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FashionRecommender:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = self._load_model()
        self.image_paths = self._get_all_images()
        self.df_embeddings = self._generate_embeddings()

    def _load_model(self):
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return model, transform

    def _get_all_images(self):
        exts = ('.jpg', '.jpeg', '.png')
        return [os.path.join(self.images_folder, f) for f in os.listdir(self.images_folder) if f.lower().endswith(exts)]

    def _extract_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return features.cpu().numpy().flatten()

    def _generate_embeddings(self):
        embeddings = []
        for path in self.image_paths:
            try:
                emb = self._extract_features(path)
                embeddings.append(emb)
            except:
                embeddings.append(np.zeros(512))
        return np.array(embeddings)

    def recommend(self, query_image_path, top_k=5):
        query_feature = self._extract_features(query_image_path).reshape(1, -1)
        similarities = cosine_similarity(query_feature, self.df_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1]
        top_indices = [i for i in top_indices if self.image_paths[i] != query_image_path][:top_k]
        return [self.image_paths[i] for i in top_indices]
