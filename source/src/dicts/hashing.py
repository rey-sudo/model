import xxhash
import struct

def string_to_coords_3d(
    text: str
) -> tuple[float, float, float]:
    """Convert a string into a deterministic 3-D coordinate.

    Strategy
    --------
    1. Encode *text* as UTF-8 and feed it to xxHash-128, which produces a
       16-byte (128-bit) digest.
    2. Split the digest into four consecutive 32-bit big-endian unsigned
       integers: ``ix``, ``iy``, ``iz``, and a spare word reserved for
       future extensions (e.g. a W component or metadata tag).

    The mapping is **purely deterministic**: identical inputs always yield
    identical outputs regardless of platform or process state.

    Parameters
    ----------
    text:
        Input string.  May be empty or contain arbitrary Unicode; it is
        always encoded as UTF-8 before hashing.

    Returns
    -------
    tuple[int, int, int]
        ``(x, y, z)`` as unsigned 32-bit integers in ``[0, 4_294_967_295]``.
    """
    # --- Step 1: hash -------------------------------------------------------
    # Encode the string as UTF-8 (handles ASCII, accented chars, emoji, etc.)
    # and compute the 128-bit xxHash digest (16 bytes).    
    digest: bytes = xxhash.xxh128(text.encode("utf-8")).digest()  # 16 bytes

    # --- Step 2: unpack four uint32 values ----------------------------------
    # ">IIII" = big-endian, four unsigned 32-bit integers.
    # The fourth word (_) is unused here but kept for forward compatibility.
    ix, iy, iz, _ = struct.unpack(">IIII", digest)

    return (ix, iy, iz)
