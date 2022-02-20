import tqdm

from src.config import Config
from src.img_processing import ScanProcessor
from src.image_handler import ImageHandler



def main() -> None:
    config = Config.from_file('cfg\\config.yml')
    proc = ScanProcessor(config.IMAGE_PROCESSING)
    handler = ImageHandler(config.IO)

    images_processed = []

    ## No new scans, exit
    if len(handler) == 0:
        input('Nothing to process or device not found. Press Enter to exit')
        return

    ## Process images with nice progress bar
    for i, (image_path, img) in tqdm.tqdm(enumerate(handler), total=len(handler), desc="Creating pdf file"):
        ip = proc(img)
        images_processed.append(ip)

    ## Save as pdf or images
    handler.save(images_processed)

    ## Update config
    new_last_idx = handler.max()
    if new_last_idx != -1:
        config.IO.LAST_IMAGE_IDX = new_last_idx
    config.save()

    input('{} pages processed. Press Enter to exit'.format(len(handler)))


if __name__ == '__main__':
    main()