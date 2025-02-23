def passwordValidation(password):
    if len(password) < 6 and len(password) > 16:
        return False
    has_lower = False
    has_upper = False
    has_digit = False
    has_symbol = False
    symbols = "$@#"
    for char in password:
        if char.islower():
            has_lower = True
        elif char.isupper():
            has_upper = True
        elif char.isdigit():
            has_digit = True
        elif char in symbols:
            has_symbol = True
    if has_lower and has_upper and has_digit and has_symbol:
        return True
    return False


# Test the function


passwords = [
    "Password123",  # Invalid (no special character)
    "Password@123",  # Valid
    "pass@123",  # Invalid (no uppercase letter)
    "PASS@123",  # Invalid (no lowercase letter)
    "Pass@12",  # Valid
    "Pass@1",  # Valid (minimum length)
    "Password@1234567",  # Invalid (too long)
    "P@1",  # Invalid (too short)
]

# Check the validity of each password


for pwd in passwords:
    result = "Valid" if passwordValidation(pwd) else "Invalid"
    print(f"Password: {pwd} - {result}")
