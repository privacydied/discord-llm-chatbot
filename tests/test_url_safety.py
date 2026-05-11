import asyncio
import pytest
from bot.url_safety import validate_url, is_private_ip, is_metadata_ip, resolve_hostname
from ipaddress import ip_address

def test_validate_url_allowed():
    # Valid HTTP/HTTPS URLs should pass
    validate_url("https://example.com")
    validate_url("http://example.com")
    validate_url("https://www.example.com/path?query=123")
    print("✅ Valid URLs pass validation")

def test_validate_url_rejects_invalid_schemes():
    # Non-HTTP/S schemes should be rejected
    from bot.exceptions import UrlSafetyError
    try:
        validate_url("ftp://example.com")
        assert False, "Should have raised UrlSafetyError"
    except UrlSafetyError as e:
        assert "scheme" in str(e).lower()
    try:
        validate_url("javascript:alert(1)")
        assert False, "Should have raised UrlSafetyError"
    except UrlSafetyError as e:
        assert "scheme" in str(e).lower()
    print("✅ Invalid schemes are rejected")

def test_validate_url_rejects_private_ips():
    # URLs resolving to private IPs should be rejected
    from bot.exceptions import UrlSafetyError
    # 127.0.0.1
    try:
        validate_url("http://127.0.0.1")
        assert False, "Should have raised UrlSafetyError for 127.0.0.1"
    except UrlSafetyError as e:
        assert "private" in str(e).lower() or "loopback" in str(e).lower()
    # 10.0.0.1
    try:
        validate_url("http://10.0.0.1")
        assert False, "Should have raised UrlSafetyError for 10.0.0.1"
    except UrlSafetyError as e:
        assert "private" in str(e).lower()
    # 192.168.1.1
    try:
        validate_url("http://192.168.1.1")
        assert False, "Should have raised UrlSafetyError for 192.168.1.1"
    except UrlSafetyError as e:
        assert "private" in str(e).lower()
    print("✅ Private IPs are rejected")

def test_validate_url_rejects_metadata_ips():
    # Metadata service IPs should be rejected
    from bot.exceptions import UrlSafetyError
    try:
        validate_url("http://169.254.169.254")
        assert False, "Should have raised UrlSafetyError for metadata IP"
    except UrlSafetyError as e:
        assert "metadata" in str(e).lower() or "169.254" in str(e).lower()
    print("✅ Metadata IPs are rejected")

def test_resolve_hostname():
    # Should resolve public hostnames to IPs
    ips = asyncio.run(resolve_hostname("example.com"))
    assert len(ips) > 0
    for ip in ips:
        assert not is_private_ip(ip)
        assert not is_metadata_ip(ip)
    print("✅ DNS resolution works correctly")

def test_is_private_ip():
    assert is_private_ip(ip_address("10.0.0.1")) is True
    assert is_private_ip(ip_address("172.16.0.0")) is True
    assert is_private_ip(ip_address("192.168.0.1")) is True
    assert is_private_ip(ip_address("127.0.0.1")) is True
    assert is_private_ip(ip_address("169.254.0.0")) is True
    assert is_private_ip(ip_address("8.8.8.8")) is False  # Google DNS is public
    print("✅ Private IP detection works")

def test_is_metadata_ip():
    assert is_metadata_ip(ip_address("169.254.169.254")) is True
    # 169.254.0.0/16 is link-local, not metadata, so this should be False
    assert is_metadata_ip(ip_address("169.254.0.0")) is False
    assert is_metadata_ip(ip_address("8.8.8.8")) is False
    print("✅ Metadata IP detection works")

if __name__ == "__main__":
    test_validate_url_allowed()
    test_validate_url_rejects_invalid_schemes()
    test_validate_url_rejects_private_ips()
    test_validate_url_rejects_metadata_ips()
    test_resolve_hostname()
    test_is_private_ip()
    test_is_metadata_ip()
    print("\nAll URL safety tests passed!")
