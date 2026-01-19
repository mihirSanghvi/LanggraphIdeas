import pytest
from backend.tools.sql import is_readonly_sql

def test_select_is_safe():
    assert is_readonly_sql("SELECT * FROM users")
    assert is_readonly_sql("select count(*) from table")

def test_nested_forbidden_words():
    # If "update" is part of a string or column name that isn't a keyword (simplistic regex check might fail this if not careful, but let's see)
    # Our regex checks for word boundaries.
    assert is_readonly_sql("SELECT * FROM updates") # Should be true because 'updates' is a table name here, but let's test our regex.
    # Actually, our regex `(^|[\s;])UPDATE([\s;]|$)` matches " UPDATE " or start "UPDATE".
    # "SELECT * FROM updates" -> "updates" is preceded by " " but followed by EOF. 
    # Wait, our regex is `r'(^|[\s;])' + word + r'([\s;]|$)'`.
    # "updates" does contain "update" but "update" is not a whole word match if we look for boundaries around 'update'.
    # But "updates" ends with 's', so it shouldn't match 'update'.
    pass

def test_forbidden_commands():
    assert not is_readonly_sql("DROP TABLE users")
    assert not is_readonly_sql("DELETE FROM users")
    assert not is_readonly_sql("TRUNCATE TABLE x")
    assert not is_readonly_sql("INSERT INTO users VALUES (1)")
    assert not is_readonly_sql("UPDATE users SET x=1")
    assert not is_readonly_sql("GRANT ALL TO user")

def test_edge_cases():
    assert not is_readonly_sql("SELECT *; DROP TABLE users")
    assert not is_readonly_sql("SELECT *; DELETE FROM users")

if __name__ == "__main__":
    # simple runner if pytest not available
    try:
        test_select_is_safe()
        test_forbidden_commands()
        test_edge_cases()
        print("All safety tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        exit(1)
