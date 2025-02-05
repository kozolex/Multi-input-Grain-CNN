import pandas as pd

def update_csv_paths(csv_path, old_path, new_path, output_csv=None):
    """
    Podmienia fragment ścieżki w kolumnie 'path' w pliku CSV i zapisuje zmodyfikowany plik.

    Args:
        csv_path (str): Ścieżka do pliku CSV.
        old_path (str): Fragment ścieżki, który ma zostać zastąpiony.
        new_path (str): Nowa ścieżka, którą należy podstawić.
        output_csv (str, optional): Ścieżka do nowego pliku CSV. Jeśli None, nadpisuje oryginalny plik.
    """
    # Wczytaj plik CSV
    df = pd.read_csv(csv_path)

    # Podmień fragment ścieżki
    df["path"] = df["path"].str.replace(old_path, new_path, regex=False)

    # Zapisz zmodyfikowany plik
    output_csv = output_csv if output_csv else csv_path
    df.to_csv(output_csv, index=False)

    print(f"Zaktualizowano ścieżki w pliku: {output_csv}")
