import os
import zipfile
import gdown
from pathlib import Path

class DataLoader:
    def __init__(self, raw_dir="data/raw/", zip_file_id="YOUR_FILE_ID_HERE"):
        """
        Initialize DataLoader with raw data directory and Google Drive file ID for 'complaints.csv.zip'.
        
        Args:
            raw_dir (str): Directory to store raw data (default: "data/raw/").
            zip_file_id (str): Google Drive file ID for 'complaints.csv.zip'. Replace 'YOUR_FILE_ID_HERE' 
                              with the ID from the shareable link (e.g., from https://drive.google.com/file/d/FILE_ID/view).
        """
        self.raw_dir = Path(raw_dir)
        self.zip_file_id = zip_file_id
        self.zip_file = self.raw_dir / "complaints.csv.zip"
        self.csv_file = self.raw_dir / "complaints.csv"

    def setup_directories(self):
        """Create raw data directory if it doesn't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_zip(self):
        """Download the 'complaints.csv.zip' file from Google Drive using the file ID."""
        if not self.zip_file.exists():
            print(f"Downloading {self.zip_file.name} from Google Drive...")
            url = f"https://drive.google.com/uc?id={self.zip_file_id}"
            try:
                gdown.download(url, str(self.zip_file), quiet=False)
                print(f"Successfully downloaded {self.zip_file.name}.")
            except Exception as e:
                raise RuntimeError(f"Failed to download {self.zip_file.name}: {str(e)}")
        else:
            print(f"{self.zip_file.name} already exists, skipping download.")

    def unzip_file(self):
        """Unzip the downloaded file and save 'complaints.csv' to the raw directory."""
        if not self.zip_file.exists():
            raise FileNotFoundError(f"{self.zip_file.name} not found. Please download it first.")

        print(f"Unzipping {self.zip_file.name}...")
        try:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                # Extract only 'complaints.csv' (adjust if zip contains multiple files)
                zip_ref.extract('complaints.csv', self.raw_dir)
            print(f"Saved {self.csv_file}.")
        except KeyError:
            raise ValueError("Zip file does not contain 'complaints.csv'. Check the file contents.")
        except Exception as e:
            raise RuntimeError(f"Failed to unzip {self.zip_file.name}: {str(e)}")

    def load_data(self):
        """Execute the full data loading pipeline: download and unzip."""
        self.setup_directories()
        self.download_zip()
        self.unzip_file()
        return self

    def get_data_path(self):
        """Return the path to the processed CSV."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"{self.csv_file} not found. Run load_data() first.")
        return self.csv_file

# Example usage
if __name__ == "__main__":
    # Replace 'YOUR_FILE_ID_HERE' with the actual Google Drive file ID for 'complaints.csv.zip'
    loader = DataLoader(zip_file_id="1xzGB5_K4IWvQOZDXGpzgrmBmfdaDek4Y")
    try:
        loader.load_data()
        print(f"Data ready at: {loader.get_data_path()}")
    except Exception as e:
        print(f"Error: {str(e)}")