import hmac
import hashlib
import time
import os
import secrets
from dataclasses import dataclass
from typing import Optional, Set, Dict

# seL4-style capability rights bitmask
CAP_READ = 0x01
CAP_WRITE = 0x02
CAP_EXEC = 0x04
CAP_CONSOLIDATE = 0x08
CAP_ALL = 0x0F


@dataclass(frozen=True)
class CapabilityToken:
    """
    Cryptographically verifiable seL4 capability token for hardware/TEE boundary isolation.
    Restricts access to enclave endpoints and shared-memory IPC rings with monotonic sequence
    counters and ephemeral lease-based revocation.
    """
    enclave_id: str
    rights: int
    nonce: str
    timestamp: float
    signature: str
    seq_num: int = 1
    lease_expiry: float = 0.0

    def has_right(self, right: int) -> bool:
        """Checks whether token holds the requested permission bit."""
        return bool(self.rights & right)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Checks whether the capability lease has expired."""
        now = current_time or time.time()
        if self.lease_expiry > 0.0:
            return now > self.lease_expiry
        return (now - self.timestamp) > 3600.0


class CapabilityManager:
    """Mints, validates, renews, and revokes capability tokens with anti-replay protection."""

    def __init__(self, master_secret: Optional[bytes] = None):
        self._secret = master_secret or secrets.token_bytes(32)
        self._revoked_nonces: Set[str] = set()
        self._latest_seq: Dict[str, int] = {}

    def mint_token(self, enclave_id: str, rights: int = CAP_ALL, lease_sec: float = 3600.0) -> CapabilityToken:
        """Issues a signed capability token for an enclave participant with lease duration."""
        nonce = secrets.token_hex(16)
        ts = time.time()
        expiry = ts + lease_sec
        seq = self._latest_seq.get(enclave_id, 0) + 1
        self._latest_seq[enclave_id] = seq

        payload = f"{enclave_id}:{rights}:{nonce}:{ts}:{seq}:{expiry}".encode("utf-8")
        sig = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        return CapabilityToken(
            enclave_id=enclave_id,
            rights=rights,
            nonce=nonce,
            timestamp=ts,
            signature=sig,
            seq_num=seq,
            lease_expiry=expiry
        )

    issue_token = mint_token

    def verify_token(
        self,
        token: CapabilityToken,
        required_right: Optional[int] = None,
        check_monotonic: bool = False
    ) -> bool:
        """Validates capability token integrity, revocation status, permissions, and leases."""
        if token.nonce in self._revoked_nonces or token.is_expired():
            return False

        # Support both new payload with seq/expiry and backward compatible tokens
        if token.lease_expiry > 0.0:
            payload = f"{token.enclave_id}:{token.rights}:{token.nonce}:{token.timestamp}:{token.seq_num}:{token.lease_expiry}".encode("utf-8")
        else:
            payload = f"{token.enclave_id}:{token.rights}:{token.nonce}:{token.timestamp}".encode("utf-8")

        expected_sig = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(token.signature, expected_sig):
            return False

        if check_monotonic:
            latest = self._latest_seq.get(token.enclave_id, 0)
            if token.seq_num < latest:
                return False

        if required_right is not None and not token.has_right(required_right):
            return False

        return True

    def renew_lease(self, token: CapabilityToken, extension_sec: float = 300.0) -> CapabilityToken:
        """Renews an active capability token lease, incrementing monotonic counter."""
        if not self.verify_token(token):
            raise PermissionError("Cannot renew invalid or expired capability token")

        # Revoke old nonce to prevent replay
        self.revoke_token(token)
        return self.mint_token(token.enclave_id, rights=token.rights, lease_sec=extension_sec)

    def revoke_token(self, token: CapabilityToken) -> None:
        """Revokes a capability token."""
        self._revoked_nonces.add(token.nonce)

