"""Bound encoded image geometry before allocating native decoded image memory."""
import struct


def jpeg_geometry(raw: bytes, max_side=1024, max_pixels=1024**2):
    if not raw.startswith(b'\xff\xd8'):
        raise ValueError('JPEG signature required.')
    offset = 2
    while offset < len(raw):
        if raw[offset] != 255:
            raise ValueError('Malformed JPEG marker.')
        while offset < len(raw) and raw[offset] == 255:
            offset += 1
        if offset >= len(raw):
            break
        marker = raw[offset]; offset += 1
        if marker in (0xD9, 0xDA):
            break
        if marker in (0x01, *range(0xD0, 0xD8)):
            continue
        if offset + 2 > len(raw):
            break
        length = struct.unpack('>H', raw[offset:offset+2])[0]
        if length < 2 or offset + length > len(raw):
            raise ValueError('Truncated JPEG segment.')
        if marker in {0xC0, 0xC1, 0xC2}:
            if length < 8:
                raise ValueError('Invalid JPEG dimensions.')
            height, width = struct.unpack('>HH', raw[offset+3:offset+7])
            if not 1 <= width <= max_side or not 1 <= height <= max_side or width * height > max_pixels:
                raise ValueError('JPEG geometry exceeds pre-decode limits.')
            return width, height
        offset += length
    raise ValueError('Supported JPEG frame header required before scan data.')
