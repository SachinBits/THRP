"""Convert Kaggle `labels_with_split.csv` into THRP hierarchical layout.

Creates directory structure: <dataset_root>/<SuperClass>/<split>/<Aircraft>/{images,labels}
Labels are written in YOLO format with class id 0 (placeholder). The existing
train_specialist/trainer code will re-index classes for specialists/generalist.
"""
import csv
from pathlib import Path
import shutil
import logging

from aircraft_superclass_map import AIRCRAFT_TO_SUPERCLASS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def map_to_superclass(aircraft: str) -> str:
    return AIRCRAFT_TO_SUPERCLASS.get(aircraft, 'Unknown')


def to_yolo(row):
    # row: filename,width,height,class,xmin,ymin,xmax,ymax,split
    w = float(row['width']); h = float(row['height'])
    xmin = float(row['xmin']); ymin = float(row['ymin'])
    xmax = float(row['xmax']); ymax = float(row['ymax'])
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def main(dataset_root: Path):
    dataset_root = Path(dataset_root)
    csv_path = dataset_root / 'labels_with_split.csv'
    src_images_dir = dataset_root / 'dataset'
    generated_top_levels = [
        'AEW&C', 'Amphibious Aircraft', 'Attack Aircraft', 'Bomber', 'Cargo', 'Drone',
        'Fighter', 'Helicopter', 'Interceptor', 'Multirole Combat', 'Prototype Aircraft',
        'Recon / Patrol', 'Reconnaissance', 'Tiltrotor', 'Transport', 'Tanker',
    ]

    if not csv_path.exists():
        logger.error('labels_with_split.csv not found in %s', dataset_root)
        return 1

    # Remove any previously generated hierarchy to avoid stale mislabeled files.
    for top in generated_top_levels:
        stale = dataset_root / top
        if stale.exists():
            shutil.rmtree(stale)

    out_root = dataset_root

    logger.info('Reading CSV: %s', csv_path)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            fname = row['filename']
            aircraft = row['class']
            split = row.get('split', 'train') or 'train'

            sc = map_to_superclass(aircraft)
            if sc == 'Unknown':
                logger.warning('Unknown superclass for %s (class=%s), assigning to Prototype Aircraft', fname, aircraft)
                sc = 'Prototype Aircraft'

            aircraft_dir_name = aircraft

            img_src = src_images_dir / (fname + '.jpg')
            if not img_src.exists():
                img_src = src_images_dir / (fname + '.jpeg')
            if not img_src.exists():
                img_src = src_images_dir / (fname + '.png')

            target_img_dir = out_root / sc / split / aircraft_dir_name / 'images'
            target_lbl_dir = out_root / sc / split / aircraft_dir_name / 'labels'
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_lbl_dir.mkdir(parents=True, exist_ok=True)

            if img_src.exists():
                dst_img = target_img_dir / img_src.name
                if not dst_img.exists():
                    shutil.copy2(img_src, dst_img)
            else:
                logger.debug('Image missing: %s', img_src)

            yolo_line = to_yolo(row)
            lbl_path = target_lbl_dir / (fname + '.txt')
            with open(lbl_path, 'a') as lf:
                lf.write(yolo_line)

            count += 1
            if count % 5000 == 0:
                logger.info('Processed %d rows', count)

    logger.info('Finished processing %d annotations', count)
    return 0


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', default='../datasets/data')
    args = p.parse_args()
    exit(main(Path(args.dataset_root)))
