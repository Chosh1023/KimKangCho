"""UID-aligned semantic raster cache, with explicit observation provenance."""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np


def validate_image_config(config):
    if config.get('schema') != 'k4-semantic-image-v1':
        raise ValueError('unsupported semantic image schema')
    if config.get('source') not in ('cvt_final_mask', 'oracle_geometry'):
        raise ValueError('image source must distinguish CVT predictions from GT geometry')
    channels = config['channels']
    if len(channels) != len(set(channels)) or len(channels) < 2:
        raise ValueError('invalid image channels')
    if config['shape'] != [200, 200] or config['bounds_m'] != [-10., 10., -10., 10.]:
        raise ValueError('first image version requires the existing 20m / 200-cell grid')
    if config.get('axes') != 'row_x_decreasing_col_y_increasing':
        raise ValueError('unknown raster axes')
    if config['valid_channel'] != channels.index('valid'):
        raise ValueError('invalid coverage channel')
    if config.get('values') != 'binary_0_1':
        raise ValueError('first version requires binary semantic masks')
    return dict(config)


class ImageCache:
    def __init__(self, root, dataset_sha256=None):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('state') != 'complete':
            raise ValueError('image cache is not complete')
        if dataset_sha256 and self.manifest['dataset_sha256'] != dataset_sha256:
            raise ValueError('image cache belongs to a different numeric dataset')
        self.config = validate_image_config(self.manifest['image_config'])
        self.index = {}
        for shard in self.manifest['shards']:
            file = (self.root/shard['file']).resolve()
            if not file.is_relative_to(self.root):
                raise ValueError('image shard path outside cache')
            for i, uid in enumerate(shard['uids']):
                if uid in self.index:
                    raise ValueError('duplicate image UID')
                self.index[uid] = (file, i, shard['split'])
        self.loaded = OrderedDict()

    def sample(self, uid, start, split=None):
        if uid not in self.index:
            raise ValueError('missing image for scene '+uid)
        file, row, stored_split = self.index[uid]
        if split is not None and split != stored_split:
            raise ValueError('image split mismatch')
        if file not in self.loaded:
            with np.load(file, allow_pickle=False) as z:
                images, starts, uids = z['images'], z['starts'], z['uids']
            expected = (len(uids), len(self.config['channels']), *self.config['shape'])
            if images.shape != expected or images.dtype != np.uint8 or images.max(initial=0) > 1:
                raise ValueError('invalid cached semantic mask')
            if starts.shape != (len(uids), 3) or not np.isfinite(starts).all():
                raise ValueError('invalid cached start poses')
            self.loaded[file] = (images, starts, uids)
            if len(self.loaded) > 2:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(file)
        images, starts, uids = self.loaded[file]
        if str(uids[row]) != uid or not np.allclose(starts[row], np.asarray(start)[:3], atol=1e-5, rtol=0):
            raise ValueError('image UID / start-pose alignment mismatch')
        return dict(bev_image=images[row].astype(np.float32), bev_start=starts[row].astype(np.float32))

    def attach_scene(self, scene, uid, split=None):
        result = dict(scene)
        result.update(self.sample(uid, scene['start_state'], split))
        result['image_config'] = self.config
        return result
