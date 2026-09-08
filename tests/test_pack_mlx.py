"""Small byte-preservation and corruption checks for the offline MLX packer."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('pack_mlx',Path(__file__).parents[1]/'scripts/pack_mlx.py')
packer=importlib.util.module_from_spec(spec);spec.loader.exec_module(packer)

class PackMLXTest(unittest.TestCase):
    def test_bytes_and_vision_filter(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe',
                'text_config':{},'quantization':{'bits':4,'group_size':64,'mode':'affine'}}))
            name='language_model.fixture.weight'
            payload=struct.pack('<4I',0,0x01234567,0x89abcdef,0xffffffff)
            header={name:{'dtype':'U32','shape':[2,2],'data_offsets':[0,16]},
                    'vision_tower.weight':{'dtype':'U32','shape':[1],'data_offsets':[16,20]}}
            encoded=json.dumps(header).encode()
            (root/'one.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload+b'xxxx')
            index={'weight_map':{name:'one.safetensors','vision_tower.weight':'one.safetensors'}}
            (root/'model.safetensors.index.json').write_text(json.dumps(index))
            out=root/'model.bin';packer.pack(root,out)
            data=out.read_bytes();length=struct.unpack('<Q',data[:8])[0];meta=json.loads(data[8:8+length])
            self.assertEqual(data[8+length:],payload)
            self.assertEqual(set(meta),{name,'__metadata__'})
            self.assertEqual(meta[name]['dtype'],'U32')
            self.assertEqual(meta[name]['shape'],[2,2])
            # A truncated tensor must never be accepted as a complete artifact.
            (root/'one.safetensors').write_bytes((root/'one.safetensors').read_bytes()[:-8])
            with self.assertRaises(ValueError):packer.pack(root,root/'bad.bin')
            self.assertFalse((root/'bad.bin').exists())

if __name__=='__main__':unittest.main()
