import gzip
import os
import urllib.request


def download_and_convert():
    # Define URLs and filenames
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz",
    }

    data_dir = os.path.dirname(__file__)

    # Download raw files
    for name, filename in files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"[PROGRESS] Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, path)

    # Conversion Logic
    def convert(img_file, label_file, out_csv, n_images):
        print(f"[PROGRESS] Creating {out_csv}...")
        img_path = os.path.join(data_dir, img_file)
        lbl_path = os.path.join(data_dir, label_file)
        out_path = os.path.join(data_dir, out_csv)

        with (
            gzip.open(img_path, "rb") as f_img,
            gzip.open(lbl_path, "rb") as f_lbl,
            open(out_path, "w") as f_out,
        ):
            # Skip headers (16 bytes for images, 8 for labels)
            f_img.read(16)
            f_lbl.read(8)

            for _ in range(n_images):
                # Read 1 byte for label, 784 bytes for pixels
                label = ord(f_lbl.read(1))
                pixels = [str(ord(f_img.read(1))) for _ in range(784)]
                f_out.write(f"{label}," + ",".join(pixels) + "\n")

        print(f"[SUCCESS] Created {out_csv}")

    # Run conversions
    convert(files["train_img"], files["train_lbl"], "mnist_train.csv", 60000)
    convert(files["test_img"], files["test_lbl"], "mnist_test.csv", 10000)


if __name__ == "__main__":
    download_and_convert()
