import os
import pandas as pd
from PIL import Image

class ImageResizer:
    def __init__(self, output_dir, target_size=(224, 224), fill_color=(0, 0, 0)):
        """
        Klasa do zmiany rozdzielczości obrazów na podstawie pliku CSV.
        
        Args:
            output_dir (str): Ścieżka do katalogu wyjściowego.
            target_size (tuple): Docelowa rozdzielczość (szerokość, wysokość).
            fill_color (tuple): Kolor uzupełnienia w przypadku zmiany proporcji (domyślnie czarny).
        """
        self.output_dir = output_dir
        self.target_size = target_size
        self.fill_color = fill_color

    def resize_and_pad(self, image):
        """
        Skaluje obraz, zachowując proporcje, a następnie dodaje tło w razie potrzeby.
        
        Args:
            image (PIL.Image): Obiekt obrazu.
            
        Returns:
            PIL.Image: Przeskalowany obraz z zachowaniem proporcji.
        """
        original_size = image.size  # (szerokość, wysokość)
        target_w, target_h = self.target_size

        # Oblicz skalę, aby zachować proporcje
        scale = min(target_w / original_size[0], target_h / original_size[1])
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))

        # Zmień rozmiar obrazu z użyciem antyaliasingu
        resized_image = image.resize(new_size, Image.LANCZOS)

        # Utwórz nowy obraz z tłem
        new_image = Image.new("RGB" if image.mode == "RGB" else "L", self.target_size, self.fill_color)
        paste_position = ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2)
        new_image.paste(resized_image, paste_position)

        return new_image

    def process_csv(self, csv_path):
        """
        Przetwarza wszystkie obrazy PNG z pliku CSV, zmieniając ich rozdzielczość.
        
        Args:
            csv_path (str): Ścieżka do pliku CSV.
        """
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_path = row["path"]
            #print(f"Przetwarzanie: {img_path}")
            #class_name = row["class"]  # Klasa obrazu
            rel_path = os.path.join(os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path))
            #print(rel_path)
            # Ścieżka do katalogu wyjściowego
            output_path = os.path.join(self.output_dir, rel_path)
            #print(output_path)
            # Tworzenie katalogu, jeśli nie istnieje
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # Wybór koloru tła na podstawie końcówki nazwy pliku
                if img_path[:-6] == ("_S.png"):
                    self.fill_color  = (255, 255, 255)  # Białe tło
                else:
                    self.fill_color  = (0, 0, 0)  # Czarne tło
                # Otwórz obraz
                image = Image.open(img_path)

                # Przetwórz obraz
                processed_image = self.resize_and_pad(image)

                # Zapisz wynik
                processed_image.save(output_path)
                #print(f"Przetworzono: {output_path}")

            except Exception as e:
                print(f"Błąd przetwarzania {img_path}: {e}")

