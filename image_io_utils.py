from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

def load_images(dir_path: str, max_workers: int | None = None) -> Tuple[List[np.ndarray], List[str]]:
    start = time.time()
    p = Path(dir_path).expanduser().resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")
    img_files = [f for f in p.iterdir() if f.suffix.lower() in {".tif", ".tiff"}]
    img_files.sort(key=lambda f: f.name)
    images: List[np.ndarray] = []
    filenames: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(cv2.imread, str(f), cv2.IMREAD_UNCHANGED): f for f in img_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading images"):
            path = futures[future]
            img = future.result()
            if img is None:
                logging.warning("Skipped corrupt image: %s", path.name)
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            filenames.append(path.name)
    print(f"load_images took {time.time() - start:.3f}s")
    out_dir=Path("processed")
    out_dir.mkdir(exist_ok=True)
    for img, name in zip(images, filenames):
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return images, filenames