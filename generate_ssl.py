"""
Generate a self-signed SSL certificate for HTTPS.
Run automatically on startup if ssl/cert.pem and ssl/key.pem don't exist.
"""
import os
import socket
import ipaddress
from datetime import datetime, timezone, timedelta
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_ssl_cert(cert_path: str = "ssl/cert.pem", key_path: str = "ssl/key.pem"):
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return  # Already exists

    os.makedirs(os.path.dirname(cert_path), exist_ok=True)
    print("Generating self-signed SSL certificate…")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    local_ip = get_local_ip()
    hostname = socket.gethostname()

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SwingTrader"),
        x509.NameAttribute(NameOID.COMMON_NAME, "swingtrader.local"),
    ])

    san_entries = [
        x509.DNSName("localhost"),
        x509.DNSName(hostname),
        x509.DNSName("swingtrader.local"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
    ]
    # Add LAN IP if different from loopback
    if local_ip not in ("127.0.0.1", "0.0.0.0"):
        try:
            san_entries.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
        except Exception:
            pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"SSL certificate generated for: localhost, {hostname}, {local_ip}")
    print(f"  Cert: {cert_path}")
    print(f"  Key:  {key_path}")
    print("  NOTE: You will need to accept the security warning in your browser.")


if __name__ == "__main__":
    generate_ssl_cert()
