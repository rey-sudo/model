import xxhash
import struct

def string_to_coords_3d(
    text: str,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-1.0, 1.0),
    z_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[float, float, float]:
    """Convert a string into a deterministic 3-D coordinate.
 
    Strategy
    --------
    1. Encode *text* as UTF-8 and feed it to xxHash-128, which produces a
       16-byte (128-bit) digest.
    2. Split the digest into four consecutive 32-bit big-endian unsigned
       integers: ``ix``, ``iy``, ``iz``, and a spare word reserved for
       future extensions (e.g. a W component or metadata tag).
    3. Normalise each integer from ``[0, MAX_UINT32]`` to ``[0, 1]``, then
       scale linearly to the requested axis range.
 
    The mapping is **purely deterministic**: identical inputs always yield
    identical outputs regardless of platform or process state.
 
    Parameters
    ----------
    text:
        Input string.  May be empty or contain arbitrary Unicode; it is
        always encoded as UTF-8 before hashing.
    x_range:
        ``(min, max)`` bounds for the X axis.  Defaults to ``(-1.0, 1.0)``.
    y_range:
        ``(min, max)`` bounds for the Y axis.  Defaults to ``(-1.0, 1.0)``.
    z_range:
        ``(min, max)`` bounds for the Z axis.  Defaults to ``(-1.0, 1.0)``.
 
    Returns
    -------
    tuple[float, float, float]
        ``(x, y, z)`` coordinates inside the specified bounding box.
    """
    # --- Step 1: hash -------------------------------------------------------
    # Encode the string as UTF-8 (handles ASCII, accented chars, emoji, etc.)
    # and compute the 128-bit xxHash digest (16 bytes).    
    digest: bytes = xxhash.xxh128(text.encode("utf-8")).digest()  # 16 bytes

    # --- Step 2: unpack four uint32 values ----------------------------------
    # ">IIII" = big-endian, four unsigned 32-bit integers.
    # The fourth word (_) is unused here but kept for forward compatibility.
    ix, iy, iz, _ = struct.unpack(">IIII", digest)

    _MAX_UINT32 = 0xFFFF_FFFF
    
    # --- Step 3: scale to target ranges -------------------------------------
    def _scale(value: int, lo: float, hi: float) -> float:
        """Linearly map *value* from ``[0, MAX_UINT32]`` to ``[lo, hi]``.
 
        Parameters
        ----------
        value:
            Raw unsigned 32-bit integer from the hash digest.
        lo:
            Lower bound of the target range.
        hi:
            Upper bound of the target range.
 
        Returns
        -------
        float
            Scaled value within ``[lo, hi]``.
        """
        return lo + (value / _MAX_UINT32) * (hi - lo)

    x = _scale(ix, *x_range)
    y = _scale(iy, *y_range)
    z = _scale(iz, *z_range)

    return (x, y, z)
