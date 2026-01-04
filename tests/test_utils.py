import pytest
from utils import enforcer, fuzzy_check

def test_clean_basic():
    raw = "| UA AA 1234 BB |"
    assert enforcer.clean(raw) == "UAAA1234BB"

def test_enforce_ua_correction():
    # Input: OCR confused '0' with 'O' and '8' with 'B'
    # Format: LL DDDD LL
    dirty_input = "VABO0001OO" 
    expected = "BO0001OO"
    assert enforcer.enforce_ua(dirty_input) == expected

def test_enforce_ua_too_short():
    short_input = "A123"
    assert enforcer.enforce_ua(short_input) == "A123"

def test_enforce_eu_cleanup():
    dirty_eu = "PL WA12345"
    assert enforcer.enforce_eu_general(dirty_eu) == "WA12345"

def test_fuzzy_check_exact_match():
    detected = "AA0055BP"
    database = "AA0055BP, KA0132CO"
    allowed, result, _ = fuzzy_check(detected, database, "Auto")
    assert allowed is True
    assert result == "AA0055BP"

def test_fuzzy_check_one_error():
    detected = "AA0O55BP" 
    database = "AA0055BP"
    allowed, result, info = fuzzy_check(detected, database, "Auto")
    
    assert allowed is True
    assert "Fuzzy fix" in info

def test_fuzzy_check_denied():
    detected = "BB9999CC"
    database = "AA0055BP"
    allowed, _, _ = fuzzy_check(detected, database, "Auto")
    assert allowed is False