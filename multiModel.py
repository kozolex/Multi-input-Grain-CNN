import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMultiInputModel(nn.Module):
    def __init__(self, num_classes=11, filter_num_base=16):
        """
        Nowy model dla obrazów o wymiarach (80, 224) zamiast (224, 224).
        
        Args:
            num_classes (int): Liczba klas do klasyfikacji.
            filter_num_base (int): Podstawowa liczba filtrów w pierwszej warstwie konwolucyjnej.
        """
        super(CustomMultiInputModel, self).__init__()

        # Konwolucyjna część modelu dla widoków RGB (T i B)
        self.rgb_model = nn.Sequential(
            nn.Conv2d(3, filter_num_base, kernel_size=3, stride=1, padding=1),  # 80x224 -> 80x224
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 80x224 -> 40x112

            nn.Conv2d(filter_num_base, filter_num_base * 2, kernel_size=3, stride=1, padding=1),  # 40x112
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 40x112 -> 20x56

            nn.Conv2d(filter_num_base * 2, filter_num_base * 4, kernel_size=3, stride=1, padding=1),  # 20x56
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 20x56 -> 10x28

            nn.Conv2d(filter_num_base * 4, filter_num_base * 8, kernel_size=3, stride=1, padding=1),  # 10x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling -> (1,1)
        )
        self.rgb_output_size = filter_num_base * 8  # 128

        # Konwolucyjna część modelu dla obrazu binarnego (S)
        self.binary_model = nn.Sequential(
            nn.Conv2d(1, filter_num_base, kernel_size=3, stride=1, padding=1),  # 80x224
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 40x112

            nn.Conv2d(filter_num_base, filter_num_base * 2, kernel_size=3, stride=1, padding=1),  # 40x112
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 20x56

            nn.Conv2d(filter_num_base * 2, filter_num_base * 4, kernel_size=3, stride=1, padding=1),  # 20x56
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 10x28

            nn.Conv2d(filter_num_base * 4, filter_num_base * 8, kernel_size=3, stride=1, padding=1),  # 10x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling -> (1,1)
        )
        self.binary_output_size = filter_num_base * 8  # 128

        # Warstwa w pełni połączona dla końcowej klasyfikacji
        self.fc = nn.Sequential(
            nn.Linear(self.rgb_output_size * 2 + self.binary_output_size, 512),  # Łączymy RGB(T, B) + S
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, t_image, b_image, s_image):
        # Przetwarzanie widoków RGB
        t_features = self.rgb_model(t_image)
        b_features = self.rgb_model(b_image)

        # Przetwarzanie obrazu binarnego
        s_features = self.binary_model(s_image)

        # Spłaszczanie wyników
        t_features = t_features.view(t_features.size(0), -1)
        b_features = b_features.view(b_features.size(0), -1)
        s_features = s_features.view(s_features.size(0), -1)

        # Połączenie cech i klasyfikacja
        combined_features = torch.cat([t_features, b_features, s_features], dim=1)
        output = self.fc(combined_features)

        return output


class MultiInputModel(nn.Module):
    def __init__(self, num_classes=11, base_model='efficientnet_v2_m', filter_num_base=8):
        super(MultiInputModel, self).__init__()
        
        # Inicjalizacja modelu RGB
        self.base_model = base_model
        if base_model == 'custom':
            self.rgb_model, self.base_model_output_size = self._initialize_rgb_custom_model(base_model)
        self.rgb_model, self.base_model_output_size = self._initialize_rgb_model(base_model)
        print(f"Model: {base_model}, base_model_output_size: {self.base_model_output_size}")

        # Inicjalizacja modelu binarnego
        self.binary_model = nn.Sequential(
            # Warstwa 1: Splot + Pooling
            nn.Conv2d(1, filter_num_base, kernel_size=3, stride=1, padding=1),  # 224x78 -> 224x78
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 224x78 -> 112x39 (połowa w obu kierunkach)
            # Warstwa 2: Splot + Pooling
            nn.Conv2d(filter_num_base, filter_num_base * 2, kernel_size=3, stride=1, padding=1),  # 112x39 -> 112x39
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 112x39 -> 56x19
            # Warstwa 3: Splot + Pooling
            nn.Conv2d(filter_num_base * 2, filter_num_base * 4, kernel_size=3, stride=1, padding=1),  # 56x19 -> 56x19
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 56x19 -> 28x9
            # Warstwa 4: Splot + Pooling
            nn.Conv2d(filter_num_base * 4, filter_num_base * 8, kernel_size=3, stride=1, padding=1),  # 28x9 -> 28x9
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Globalne wyciągnięcie cech do 1x1
            # Spłaszczenie + w pełni połączona warstwa
            nn.Flatten(),
            nn.Linear(filter_num_base * 8, 128),  # Dopasowanie wyjścia do 128
            nn.ReLU()
        )
        self.binary_model_output_size = 128


        # Warstwa łącząca
        total_input_size = self.base_model_output_size * 2 + self.binary_model_output_size
        print(f"Total input size to fc: {total_input_size}")
        self.fc = nn.Sequential(
            nn.Linear(total_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _initialize_rgb_model(self, base_model):
        """
        Inicjalizuje wybrany model sieci RGB i zwraca model oraz rozmiar jego wyjścia.
        """
        if base_model.startswith('efficientnet'):  # Obsługa EfficientNet i EfficientNetV2
            model = getattr(models, base_model)(pretrained=True)
            model.classifier = nn.Identity()
            if base_model.startswith('efficientnet_v2'):
                return model, 1280  # Wyjście dla EfficientNetV2-M
            return model, 1280  # Wyjście dla EfficientNet-B0/B1
        
        elif base_model == 'googlenet':
            model = models.googlenet(pretrained=True)
            model.fc = nn.Identity()
            return model, 1024
        
        elif base_model == 'inception_v3':
            model = models.inception_v3(pretrained=True, aux_logits=False)  # Wyłącz dodatkowe głowice
            model.fc = nn.Identity()
            return model, 2048
        
        elif base_model == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier = nn.Identity()
            return model, 1280
        
        elif base_model == 'mobilenet_v3_large' or base_model == 'mobilenet_v3_small':
            model = getattr(models, base_model)(pretrained=True)
            model.classifier = nn.Identity()
            return model, 576
        
        elif base_model.startswith('resnet'):  # Obsługa ResNet (np. resnet18, resnet50)
            model = getattr(models, base_model)(pretrained=True)
            model.fc = nn.Identity()
            return model, 2048 if '50' in base_model or '101' in base_model else 512  # Rozmiar zależny od wariantu
        
        elif base_model == 'swin_t':
            model = models.swin_t(pretrained=True)
            model.head = nn.Identity()
            return model, 768
        
        elif base_model == 'vit_b_16':  # VisionTransformer
            model = models.vit_b_16(pretrained=True)
            model.heads = nn.Identity()
            return model, 768

        else:
            raise ValueError(f"Unsupported base model: {base_model}")
    
    def _initialize_rgb_custom_model(self, base_model):
        """
        Inicjalizuje wybrany model sieci RGB i zwraca model oraz rozmiar jego wyjścia.
        """
        if base_model.startswith('efficientnet'):  # Obsługa EfficientNet i EfficientNetV2
            model = getattr(models, base_model)(pretrained=True)
            model.classifier = nn.Identity()
            if base_model.startswith('efficientnet_v2'):
                return model, 1280  # Wyjście dla EfficientNetV2-M
            return model, 1280  # Wyjście dla EfficientNet-B0/B1
        
        elif base_model == 'googlenet':
            model = models.googlenet(pretrained=True)
            model.fc = nn.Identity()
            return model, 1024
        
        elif base_model == 'inception_v3':
            model = models.inception_v3(pretrained=True, aux_logits=False)  # Wyłącz dodatkowe głowice
            model.fc = nn.Identity()
            return model, 2048
        
        elif base_model == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier = nn.Identity()
            return model, 1280
        
        elif base_model == 'mobilenet_v3_large' or base_model == 'mobilenet_v3_small':
            model = getattr(models, base_model)(pretrained=True)
            model.classifier = nn.Identity()
            return model, 576
        
        elif base_model.startswith('resnet'):  # Obsługa ResNet (np. resnet18, resnet50)
            model = getattr(models, base_model)(pretrained=True)
            model.fc = nn.Identity()
            return model, 2048 if '50' in base_model or '101' in base_model else 512  # Rozmiar zależny od wariantu
        
        elif base_model == 'swin_t':
            model = models.swin_t(pretrained=True)
            model.head = nn.Identity()
            return model, 768
        
        elif base_model == 'vit_b_16':  # VisionTransformer
            model = models.vit_b_16(pretrained=True)
            model.heads = nn.Identity()
            return model, 768

        else:
            raise ValueError(f"Unsupported base model: {base_model}")

    def forward(self, t_image, b_image, s_image):
        # Ekstrakcja cech dla widoków RGB
        t_features = self.rgb_model(t_image)  # Widok T
        b_features = self.rgb_model(b_image)  # Widok B
        #print(f"T Features: {t_features.shape}, B Features: {b_features.shape}")

        # Ekstrakcja cech dla obrazu binarnego
        s_features = self.binary_model(s_image)
        #print(f"S Features: {s_features.shape}")

        # Połączenie cech
        combined_features = torch.cat([t_features, b_features, s_features], dim=1)
        #print(f"Combined Features: {combined_features.shape}")

        # Klasyfikacja
        output = self.fc(combined_features)
        return output

    @staticmethod
    def get_input_size(base_model):
        """
        Zwraca wymagane wymiary wejściowe dla danego modelu.
        
        Args:
            base_model (str): Nazwa modelu bazowego.
            
        Returns:
            tuple: Wymiary wejściowe modelu (wysokość, szerokość).
        """
        if base_model.startswith('efficientnet') or base_model.startswith('mobilenet'):
            return (224, 224)  # EfficientNet, MobileNet wymagają 224x224
            
        elif base_model == 'googlenet':
            return (224, 224)  # GoogLeNet wymaga 224x224
        
        elif base_model == 'inception_v3':
            return (299, 299)  # Inception V3 wymaga 299x299
        
        elif base_model == 'maxvit_t':
            return (224, 224)  # MaxVit wymaga 224x224
        
        elif base_model.startswith('resnet'):
            return (224, 224)  # ResNet (np. ResNet50/ResNet101) wymaga 224x224
        
        elif base_model.startswith('squeezenet'):
            return (224, 224)  # SqueezeNet wymaga 224x224
        
        elif base_model == 'swin_t':
            return (224, 224)  # SwinTransformer wymaga 224x224
        
        elif base_model == 'vit_b_16':  # VisionTransformer
            return (224, 224)  # VisionTransformer wymaga 224x224
        
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
    def forward2(self, t_image, b_image, s_image):
        # Pobierz wymagany rozmiar wejściowy
        input_size = self.get_input_size(self.base_model)
        
        # Weryfikacja wejścia `t_image` i `b_image` (RGB) oraz `s_image` (binary)
        assert t_image.shape[-2:] == input_size, f"Expected T image to be of size {input_size}, but got {t_image.shape[-2:]}"
        assert b_image.shape[-2:] == input_size, f"Expected B image to be of size {input_size}, but got {b_image.shape[-2:]}"
        assert s_image.shape[-2:] == input_size, f"Expected S image to be of size {input_size}, but got {s_image.shape[-2:]}"
        
        # Ekstrakcja cech dla widoków RGB
        t_features = self.rgb_model(t_image)  # Widok T
        b_features = self.rgb_model(b_image)  # Widok B

        # Ekstrakcja cech dla obrazu binarnego
        s_features = self.binary_model(s_image)

        # Połączenie cech
        combined_features = torch.cat([t_features, b_features, s_features], dim=1)

        # Klasyfikacja
        output = self.fc(combined_features)
        return output
    
class MultiInputDataset(Dataset):
    def __init__(self, csv_file, transform_rgb=None, transform_binary=None):
        self.data = pd.read_csv(csv_file)

        # Tworzenie mapowania nazw klas na liczby całkowite
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.data['class'].unique())}

        self.transform_rgb = transform_rgb
        self.transform_binary = transform_binary

    def __len__(self):
        return len(self.data) // 3  # Każde ziarno ma 3 obrazy

    def __getitem__(self, idx):
        # Pobierz trzy obrazy
        base_idx = idx * 3
        t_path = self.data.iloc[base_idx]['path']
        b_path = self.data.iloc[base_idx + 1]['path']
        s_path = self.data.iloc[base_idx + 2]['path']

        t_image = Image.open(t_path).convert("RGB")
        b_image = Image.open(b_path).convert("RGB")
        s_image = Image.open(s_path).convert("L")  # Obraz binarny

        # Transformacje
        if self.transform_rgb:
            t_image = self.transform_rgb(t_image)
            b_image = self.transform_rgb(b_image)
        if self.transform_binary:
            s_image = self.transform_binary(s_image)

        # Pobierz nazwę klasy i przekształć na indeks numeryczny
        class_name = self.data.iloc[base_idx]['class']
        label = self.class_to_idx[class_name]  # Mapowanie nazwy klasy na numer
        label = torch.tensor(label, dtype=torch.long)  # Konwersja na tensor PyTorch

        return t_image, b_image, s_image, label

#Krok 2: Transformacje dla obrazów RGB i binarnych:
# Transformacje dla obrazów RGB
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformacje dla obrazów binarnych
transform_binary = transforms.Compose([
    transforms.Resize((224 , 78)),
    transforms.ToTensor()
])

transform_rgb_224 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformacje dla obrazów binarnych
transform_binary_224 = transforms.Compose([transforms.ToTensor()])

# Transformacje dla obrazów RGB z auugumentacją dla 80x224
train_transform_rgb_80 = transforms.Compose([
    transforms.Resize((80, 224)),  # Skalowanie do 80x224
    transforms.RandomHorizontalFlip(p=0.5),  # Losowe odbicie poziome
    transforms.RandomVerticalFlip(p=0.5),  # Losowe odbicie pionowe
    transforms.RandomRotation(degrees=5),  # Losowy obrót o ±15 stopni
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Losowe zmiany jasności, kontrastu, nasycenia
    transforms.ToTensor(),  # Konwersja na tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja ImageNet
])
val_transform_rgb_80 = transforms.Compose([
    transforms.Resize((80, 224)),  # Skalowanie do 80x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Transformacje dla obrazów binarnych
transform_binary_80 = transforms.Compose([transforms.ToTensor()])

#TEST MODELU
def test_model(path_model, test_loader, device="cuda"):
    """
    Testuje model na zbiorze testowym i wyznacza macierz pomyłek.

    Args:
        path_model (str): Ścieżka do pliku .pth z zapisanymi wagami modelu.
        test_loader (DataLoader): DataLoader dla zbioru testowego.
        device (str): Urządzenie ("cuda" lub "cpu").

    Returns:
        cm: Macierz pomyłek.
        y_true: Rzeczywiste etykiety.
        y_pred: Przewidywane etykiety.
    """

    # Załaduj wagi modelu
    model = torch.load(path_model, map_location=device)
    model.eval()  # Ustawienie modelu w tryb ewaluacji

    y_true = []
    y_pred = []

    with torch.no_grad():
        for t_image, b_image, s_image, labels in test_loader:
            t_image, b_image, s_image, labels = (
                t_image.to(device),
                b_image.to(device),
                s_image.to(device),
                labels.to(device)
            )

            # Oblicz predykcje
            outputs = model(t_image, b_image, s_image)
            _, predicted = torch.max(outputs, 1)

            # Zbierz rzeczywiste i przewidywane etykiety
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Wyznaczenie macierzy pomyłek
    cm = confusion_matrix(y_true, y_pred)

    return cm, y_true, y_pred