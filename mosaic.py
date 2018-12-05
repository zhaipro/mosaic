import sys
import os

from PIL import Image

# Change these 3 config parameters to suit your needs...
TILE_SIZE      = 50     # height/width of mosaic tiles in pixels
TILE_MATCH_RES = 5      # tile matching resolution (higher values give better fit but require more processing)
ENLARGEMENT    = 8      # the mosaic image will be this many times wider and taller than the original

TILE_BLOCK_SIZE = int(TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1))
OUT_FILE = 'mosaic.jpg'


def get_tile(fn):
    try:
        img = Image.open(fn)
        # tiles must be square, so get the largest square that fits inside the image
        w, h = img.size
        min_dimension = min(w, h)
        w_crop = (w - min_dimension) / 2
        h_crop = (h - min_dimension) / 2
        img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

        large_tile_img = img.resize((TILE_SIZE, TILE_SIZE), Image.ANTIALIAS)
        small_tile_img = img.resize((int(TILE_SIZE / TILE_BLOCK_SIZE), int(TILE_SIZE / TILE_BLOCK_SIZE)), Image.ANTIALIAS)

        return large_tile_img.convert('RGB'), small_tile_img.convert('RGB')
    except:
        return None, None


def get_tiles(tiles_directory):
    large_tiles = []
    small_tiles = []

    # search the tiles directory recursively
    for root, _, files in os.walk(tiles_directory):
        for fn in files:
            tile_path = os.path.join(root, fn)
            large_tile, small_tile = get_tile(tile_path)
            if large_tile:
                large_tiles.append(large_tile)
                small_tiles.append(small_tile)

    return large_tiles, small_tiles


def get_target_image_data(fn):
    img = Image.open(fn)
    w = img.size[0] * ENLARGEMENT
    h = img.size[1] * ENLARGEMENT
    large_img = img.resize((w, h), Image.ANTIALIAS)
    w_diff = (w % TILE_SIZE)/2
    h_diff = (h % TILE_SIZE)/2

    # if necesary, crop the image slightly so we use a whole number of tiles horizontally and vertically
    print(w_diff, h_diff)
    if w_diff or h_diff:
        large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

    small_img = large_img.resize((int(w / TILE_BLOCK_SIZE), int(h / TILE_BLOCK_SIZE)), Image.ANTIALIAS)

    image_data = large_img.convert('RGB'), small_img.convert('RGB')
    return image_data


class TileFitter:
    def __init__(self, tiles_data):
        self.tiles_data = tiles_data

    def __get_tile_diff(self, t1, t2, bail_out_value):
        diff = 0
        for i in range(len(t1)):
            diff += ((t1[i][0] - t2[i][0])**2 + (t1[i][1] - t2[i][1])**2 + (t1[i][2] - t2[i][2])**2)
            if diff > bail_out_value:
                # we know already that this isnt going to be the best fit, so no point continuing with this tile
                return diff
        return diff

    def get_best_fit_tile(self, img_data):
        best_fit_tile_index = None
        min_diff = sys.maxsize
        tile_index = 0

        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        for tile_data in self.tiles_data:
            diff = self.__get_tile_diff(img_data, tile_data, min_diff)
            if diff < min_diff:
                min_diff = diff
                best_fit_tile_index = tile_index
            tile_index += 1

        return best_fit_tile_index


class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new(original_img.mode, original_img.size)
        self.x_tile_count = original_img.size[0] // TILE_SIZE
        self.y_tile_count = original_img.size[1] // TILE_SIZE

    def add_tile(self, tile_data, coords):
        img = Image.new('RGB', (TILE_SIZE, TILE_SIZE))
        img.putdata(tile_data)
        self.image.paste(img, coords)

    def save(self, path):
        self.image.save(path)


def build_mosaic(result_queue, all_tile_data_large, original_img_large):
    mosaic = MosaicImage(original_img_large)

    for img_coords, best_fit_tile_index in result_queue:
        tile_data = all_tile_data_large[best_fit_tile_index]
        mosaic.add_tile(tile_data, img_coords)

    mosaic.save(OUT_FILE)


def compose(original_img, tiles):
    original_img_large, original_img_small = original_img
    tiles_large, tiles_small = tiles

    mosaic = MosaicImage(original_img_large)

    all_tile_data_large = list(map(lambda tile: list(tile.getdata()), tiles_large))
    all_tile_data_small = list(map(lambda tile: list(tile.getdata()), tiles_small))

    # start the worker processes that will perform the tile fitting
    result_queue = []
    # this function gets run by the worker processes, one on each CPU core
    tile_fitter = TileFitter(all_tile_data_small)
    for x in range(mosaic.x_tile_count):
        for y in range(mosaic.y_tile_count):
            large_box = x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE
            small_box = x * TILE_SIZE/TILE_BLOCK_SIZE, y * TILE_SIZE/TILE_BLOCK_SIZE, (x + 1) * TILE_SIZE/TILE_BLOCK_SIZE, (y + 1) * TILE_SIZE/TILE_BLOCK_SIZE
            img_data, img_coords = list(original_img_small.crop(small_box).getdata()), large_box
            tile_index = tile_fitter.get_best_fit_tile(img_data)
            result_queue.append((img_coords, tile_index))

    # start the worker processes that will build the mosaic image
    build_mosaic(result_queue, all_tile_data_large, original_img_large)


def mosaic(img_path, tiles_path):
    tiles_data = get_tiles(tiles_path)
    image_data = get_target_image_data(img_path)
    compose(image_data, tiles_data)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <image> <tiles directory>\r')
    else:
        mosaic(sys.argv[1], sys.argv[2])
