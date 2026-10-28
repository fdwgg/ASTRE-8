"""Bounded, seek-based authored-loop metadata reader (WAV, FLAC, Vorbis)."""
from dataclasses import dataclass
from pathlib import Path
import struct
import time
from astre8_detector import checkpoint

MAX_BYTES = 256 * 1024


@dataclass(frozen=True)
class EmbeddedLoop:
    start: int
    end: int
    rate: int
    format: str


def _valid(condition):
    if not condition:
        raise ValueError("Invalid loop metadata")


def _u32(data, offset=0):
    return struct.unpack_from("<I", data, offset)[0]


class _Source:
    def __init__(self, stream, cancelled):
        self.stream, self.cancelled = stream, cancelled
        self.length = stream.seek(0, 2)
        self.steps = self.read_bytes = 0
        self.started = time.monotonic()

    def read(self, position, count):
        checkpoint(self.cancelled)
        self.steps += 1
        _valid(self.steps <= 512 and time.monotonic() - self.started < .25)
        _valid(0 <= position <= self.length and 0 <= count <= self.length - position)
        _valid(count <= MAX_BYTES - self.read_bytes)
        self.stream.seek(position)
        data = self.stream.read(count)
        _valid(len(data) == count)
        self.read_bytes += count
        return data


def _comments(data, start, framing, source):
    position = start

    def count():
        nonlocal position
        _valid(position + 4 <= len(data))
        number = _u32(data, position)
        position += 4
        _valid(number <= len(data) - position)
        return number

    vendor_length = count()
    position += vendor_length
    fields = count()
    _valid(fields <= 4096 and fields <= (len(data) - position) // 4)
    values = {}
    for i in range(fields):
        if i % 32 == 0:
            checkpoint(source.cancelled)
        size = count()
        field = data[position:position + size].decode("utf-8", errors="strict")
        position += size
        key, sep, value = field.partition("=")
        key, value = key.upper().replace("_", ""), value.strip()
        if not sep or key not in {"LOOPSTART", "LOOPEND", "LOOPLENGTH", "LOOPTYPE", "LOOPMODE"}:
            continue
        if key in {"LOOPTYPE", "LOOPMODE"}:
            _valid(value.lower() in {"0", "forward"})
        else:
            _valid(0 < len(value) <= 19 and value.isascii() and value.isdecimal())
            number = int(value)
            _valid(number <= 2**63 - 1 and (key not in values or values[key] == number))
            values[key] = number
    _valid(position == len(data) - 1 and data[position] & 1 if framing else position == len(data))
    return values


def _tag_loop(tags, rate, name):
    start, end, length = tags.get("LOOPSTART"), tags.get("LOOPEND"), tags.get("LOOPLENGTH")
    if start is None or (end is None and length is None):
        return None
    if length is not None:
        _valid(length > 0 and start + length <= 2**63 - 1 and (end is None or end == start + length))
    return EmbeddedLoop(start, end if end is not None else start + length, rate, name)


def _wave(s, frames):
    header = s.read(0, 12)
    _valid(header[8:] == b"WAVE")
    end = _u32(header, 4) + 8
    _valid(12 <= end <= s.length)
    position, rate, align, data_frames, loop = 12, 0, 0, None, None
    while position < end:
        chunk = s.read(position, 8)
        size, payload = _u32(chunk, 4), position + 8
        _valid(size <= end - payload)
        if chunk[:4] == b"fmt ":
            _valid(rate == 0 and size >= 16)
            fmt, channels, rate, _, align, bits = struct.unpack("<HHIIHH", s.read(payload, 16))
            _valid(fmt in (1, 3) and rate > 0 and channels > 0 and 8 <= bits <= 64
                   and bits % 8 == 0 and align == channels * (bits // 8))
        elif chunk[:4] == b"data":
            _valid(data_frames is None and align > 0 and size % align == 0)
            data_frames = size // align
        elif chunk[:4] == b"smpl":
            _valid(loop is None and size >= 60)
            smpl = s.read(payload, 60)
            _valid(_u32(smpl, 28) == 1 and _u32(smpl, 40) == 0 and _u32(smpl, 52) == 0
                   and _u32(smpl, 32) <= size - 60)
            loop = (_u32(smpl, 44), _u32(smpl, 48) + 1)
        padded = size + (size & 1)
        _valid(padded <= end - payload)
        position = payload + padded
    _valid(data_frames == frames and rate > 0)
    return EmbeddedLoop(*loop, rate, "WAV smpl") if loop else None


def _flac(s, frames):
    position, rate, tags, first = 4, 0, None, True
    while True:
        header = s.read(position, 4)
        kind, last, size = header[0] & 127, bool(header[0] & 128), int.from_bytes(header[1:], "big")
        payload = position + 4
        _valid(payload + size <= s.length and (not first or kind == 0))
        if kind == 0:
            _valid(first and size == 34)
            packed = int.from_bytes(s.read(payload, 34)[10:18], "big")
            rate, stream_frames = packed >> 44, packed & 0xfffffffff
            _valid(stream_frames in (0, frames))
        elif kind == 4:
            _valid(tags is None and size <= MAX_BYTES)
            tags = _comments(s.read(payload, size), 0, False, s)
        elif kind == 127:
            raise ValueError("Invalid FLAC metadata block")
        first, position = False, payload + size
        if last:
            break
    _valid(rate > 0)
    return _tag_loop(tags, rate, "FLAC comments") if tags is not None else None


def _ogg_crc(data):
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04c11db7 if crc & 0x80000000 else 0)) & 0xffffffff
    return crc


def _vorbis(s):
    position = sequence = packets = rate = 0
    serial, partial, packet = None, False, bytearray()
    while packets < 2:
        header = s.read(position, 27)
        flags = header[5]
        _valid(header[:5] == b"OggS\0" and flags & 0xf8 == 0 and bool(flags & 1) == partial
               and (sequence == 0) == bool(flags & 2))
        page_serial = _u32(header, 14)
        if serial is None:
            serial = page_serial
        _valid(page_serial == serial and _u32(header, 18) == sequence)
        sequence += 1
        lacing = s.read(position + 27, header[26])
        body = s.read(position + 27 + len(lacing), sum(lacing))
        _valid(_ogg_crc(header[:22] + bytes(4) + header[26:] + lacing + body) == _u32(header, 22))
        offset = 0
        for size in lacing:
            _valid(len(packet) + size <= MAX_BYTES)
            packet.extend(body[offset:offset + size])
            offset += size
            partial = size == 255
            if not partial:
                data, packet = bytes(packet), bytearray()
                if packets == 0:
                    _valid(len(data) == 30 and data[:7] == b"\x01vorbis" and _u32(data, 7) == 0
                           and data[11] > 0 and data[29] & 1 and offset == len(body))
                    rate = _u32(data, 12)
                else:
                    _valid(len(data) >= 8 and data[:7] == b"\x03vorbis")
                    return _tag_loop(_comments(data, 7, True, s), rate, "Vorbis comments")
                packets += 1
        _valid(flags & 4 == 0)
        position += 27 + len(lacing) + len(body)
    return None


def read_embedded_loop(path, rate, frames, cancelled=lambda: False):
    """Return validated sample coordinates, or None for absent/invalid metadata.

    Cancellation propagates. Compressed MP3/Opus loop tags remain unsupported.
    """
    checkpoint(cancelled)
    if rate <= 0 or frames <= 0:
        return None
    try:
        with Path(path).open("rb") as stream:
            s = _Source(stream, cancelled)
            signature = s.read(0, 4)
            loop = _wave(s, frames) if signature == b"RIFF" else _flac(s, frames) if signature == b"fLaC" else _vorbis(s) if signature == b"OggS" else None
            return loop if loop and loop.rate == rate and 0 <= loop.start < loop.end <= frames else None
    except (OSError, ValueError, UnicodeError, struct.error, IndexError):
        return None
