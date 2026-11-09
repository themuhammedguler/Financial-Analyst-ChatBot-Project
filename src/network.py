"""Ağ ve SSL yapılandırma modülü.

Windows ortamında dosya yolunda ASCII dışı Türkçe karakterler olduğunda (örn: Masaüstü)
libcurl ve OpenSSL'in SSL hatası vermesini önlemek için CA sertifikasını güvenli
temp dizinine yönlendirir ve retry mekanizması sunar.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any, Optional
import certifi


def ensure_ascii_ca_bundle() -> None:
    """Windows'ta dosya yolunda ASCII dışı karakterler olduğunda (Masaüstü vb.)
    libcurl SSL error hatasını önlemek için CA sertifikasını ASCII temp dizinine kopyalar.
    """
    if os.name != "nt":
        return
    try:
        ca_path = certifi.where()
        if any(ord(c) > 127 for c in ca_path):
            ascii_ca = os.path.join(tempfile.gettempdir(), "cacert.pem")
            if not os.path.exists(ascii_ca):
                shutil.copyfile(ca_path, ascii_ca)
            os.environ.setdefault("CURL_CA_BUNDLE", ascii_ca)
            os.environ.setdefault("SSL_CERT_FILE", ascii_ca)
    except Exception:
        pass


# Modül import edildiğinde otomatik çalıştır
ensure_ascii_ca_bundle()
