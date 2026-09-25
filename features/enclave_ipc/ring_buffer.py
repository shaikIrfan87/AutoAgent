import mmap
import struct
import threading
import time
from typing import Optional, List, Dict, Any, Tuple
from .capabilities import CapabilityToken, CapabilityManager, CAP_READ, CAP_WRITE


class SharedMemoryIPCRing:
    """
    Zero-copy bounded circular ring buffer modeling hardware shared-memory IPC endpoints
    between untrusted external network adapters and the seL4/TEE Trusted Computing Base (TCB).
    Enforces capability token access checks before allowing slot read/write operations.
    """

    def __init__(self, capacity: int = 64, cap_manager: Optional[CapabilityManager] = None):
        self.capacity = capacity
        self.cap_manager = cap_manager or CapabilityManager()
        self._buffer: List[Optional[Dict[str, Any]]] = [None] * capacity
        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._count = 0
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def is_full(self) -> bool:
        with self._lock:
            return self._count == self.capacity

    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    def push(self, data: bytes, token: CapabilityToken) -> int:
        """
        Pushes a byte payload into the ring buffer.
        Requires valid capability token with CAP_WRITE permission.
        Returns the slot index written to.
        """
        if not self.cap_manager.verify_token(token, required_right=CAP_WRITE):
            raise PermissionError("seL4 IPC Error: Insufficient capability rights for CAP_WRITE")

        with self._lock:
            if self._count == self.capacity:
                raise BufferError("seL4 IPC Error: Shared-memory ring buffer overflow")

            slot_idx = self._head
            self._buffer[slot_idx] = {
                "data": data,
                "sender": token.enclave_id,
                "timestamp": time.time(),
            }
            self._head = (self._head + 1) % self.capacity
            self._count += 1
            return slot_idx

    def pop(self, token: CapabilityToken) -> Optional[bytes]:
        """
        Pops the next byte payload from the ring buffer.
        Requires valid capability token with CAP_READ permission.
        """
        if not self.cap_manager.verify_token(token, required_right=CAP_READ):
            raise PermissionError("seL4 IPC Error: Insufficient capability rights for CAP_READ")

        with self._lock:
            if self._count == 0:
                return None

            entry = self._buffer[self._tail]
            self._buffer[self._tail] = None
            self._tail = (self._tail + 1) % self.capacity
            self._count -= 1
            return entry["data"] if entry else None


class BinaryMmapIPCRing:
    """
    OS Memory-Mapped (mmap) Ring Buffer for hardware-grade zero-copy IPC.
    Uses binary packed headers and fixed-size slots in shared memory.
    """
    MAGIC = b"SE4R"
    HEADER_FMT = "=4sIIIIIQ"  # magic(4s), capacity(I), slot_size(I), head(I), tail(I), count(I), seq(Q)
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    def __init__(self, capacity: int = 32, slot_size: int = 4096, cap_manager: Optional[CapabilityManager] = None):
        self.capacity = capacity
        self.slot_size = slot_size
        self.cap_manager = cap_manager or CapabilityManager()
        self.total_size = self.HEADER_SIZE + (capacity * slot_size)
        self.mm = mmap.mmap(-1, self.total_size)
        self._lock = threading.Lock()
        self._init_header()

    def _init_header(self) -> None:
        struct.pack_into(
            self.HEADER_FMT, self.mm, 0,
            self.MAGIC, self.capacity, self.slot_size, 0, 0, 0, 0
        )

    def _read_header(self) -> Tuple[int, int, int, int]:
        _, _, _, head, tail, count, seq = struct.unpack_from(self.HEADER_FMT, self.mm, 0)
        return head, tail, count, seq

    def _write_pointers(self, head: int, tail: int, count: int, seq: int) -> None:
        struct.pack_into(
            self.HEADER_FMT, self.mm, 0,
            self.MAGIC, self.capacity, self.slot_size, head, tail, count, seq
        )

    def push(self, data: bytes, token: CapabilityToken) -> int:
        if not self.cap_manager.verify_token(token, required_right=CAP_WRITE):
            raise PermissionError("seL4 IPC Error: Insufficient capability rights for CAP_WRITE")
        if len(data) > self.slot_size - 4:
            raise ValueError(f"Payload size {len(data)} exceeds slot capacity {self.slot_size - 4}")

        with self._lock:
            head, tail, count, seq = self._read_header()
            if count >= self.capacity:
                raise BufferError("seL4 IPC Error: Shared-memory ring buffer overflow")

            slot_offset = self.HEADER_SIZE + (head * self.slot_size)
            struct.pack_into("=I", self.mm, slot_offset, len(data))
            self.mm[slot_offset + 4 : slot_offset + 4 + len(data)] = data

            new_head = (head + 1) % self.capacity
            self._write_pointers(new_head, tail, count + 1, seq + 1)
            return head

    def pop(self, token: CapabilityToken) -> Optional[bytes]:
        if not self.cap_manager.verify_token(token, required_right=CAP_READ):
            raise PermissionError("seL4 IPC Error: Insufficient capability rights for CAP_READ")

        with self._lock:
            head, tail, count, seq = self._read_header()
            if count == 0:
                return None

            slot_offset = self.HEADER_SIZE + (tail * self.slot_size)
            (payload_len,) = struct.unpack_from("=I", self.mm, slot_offset)
            data = bytes(self.mm[slot_offset + 4 : slot_offset + 4 + payload_len])

            new_tail = (tail + 1) % self.capacity
            self._write_pointers(head, new_tail, count - 1, seq)
            return data

    def close(self) -> None:
        if not self.mm.closed:
            self.mm.close()
