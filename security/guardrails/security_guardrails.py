"""
Security GuardRails for Financial Audit System
"""

import re
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta


class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityViolation:
    """Represents a security violation"""
    violation_type: str
    severity: SecurityLevel
    description: str
    location: str
    remediation: str
    timestamp: datetime


class DataClassificationGuard:
    """Data classification and protection guardrails"""

    # Patterns for sensitive data detection
    PATTERNS = {
        'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'bank_account': r'\b\d{8,17}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}-?\d{3}-?\d{4}\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'api_key': r'\b[A-Za-z0-9]{32,}\b',
        'aws_key': r'\bAKIA[0-9A-Z]{16}\b',
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def scan_text(self, text: str, context: str = "") -> List[SecurityViolation]:
        """Scan text for sensitive data patterns"""
        violations = []

        for data_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                violation = SecurityViolation(
                    violation_type=f"sensitive_data_{data_type}",
                    severity=self._get_severity(data_type),
                    description=f"Detected {data_type} in {context or 'text'}",
                    location=f"Position {match.start()}-{match.end()}",
                    remediation=f"Mask or encrypt {data_type} data",
                    timestamp=datetime.now()
                )
                violations.append(violation)

        return violations

    def _get_severity(self, data_type: str) -> SecurityLevel:
        """Get severity level for data type"""
        critical_types = ['ssn', 'credit_card', 'bank_account', 'api_key', 'aws_key']
        high_types = ['email', 'phone']

        if data_type in critical_types:
            return SecurityLevel.CRITICAL
        elif data_type in high_types:
            return SecurityLevel.HIGH
        else:
            return SecurityLevel.MEDIUM

    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text"""
        masked_text = text

        for data_type, pattern in self.PATTERNS.items():
            if data_type in ['ssn', 'credit_card', 'bank_account']:
                # Full masking for highly sensitive data
                masked_text = re.sub(pattern, '***MASKED***', masked_text, flags=re.IGNORECASE)
            elif data_type in ['email', 'phone']:
                # Partial masking
                def partial_mask(match):
                    value = match.group()
                    if '@' in value:  # Email
                        local, domain = value.split('@')
                        return f"{local[:2]}***@{domain}"
                    else:  # Phone
                        return f"{value[:3]}-***-{value[-4:]}"

                masked_text = re.sub(pattern, partial_mask, masked_text, flags=re.IGNORECASE)

        return masked_text


class InputValidationGuard:
    """Input validation and sanitization guardrails"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_financial_data(self, data: Dict[str, Any]) -> List[SecurityViolation]:
        """Validate financial data input"""
        violations = []

        # Check required fields
        required_fields = ['amount', 'transaction_date', 'account_id']
        for field in required_fields:
            if field not in data or data[field] is None:
                violations.append(SecurityViolation(
                    violation_type="missing_required_field",
                    severity=SecurityLevel.HIGH,
                    description=f"Missing required field: {field}",
                    location=f"Field: {field}",
                    remediation=f"Provide valid {field}",
                    timestamp=datetime.now()
                ))

        # Validate amount
        if 'amount' in data:
            amount = data['amount']
            if not isinstance(amount, (int, float)) or amount < 0:
                violations.append(SecurityViolation(
                    violation_type="invalid_amount",
                    severity=SecurityLevel.MEDIUM,
                    description="Amount must be a positive number",
                    location="Field: amount",
                    remediation="Provide valid positive amount",
                    timestamp=datetime.now()
                ))

            if amount > 10_000_000:  # $10M threshold
                violations.append(SecurityViolation(
                    violation_type="suspicious_amount",
                    severity=SecurityLevel.HIGH,
                    description="Amount exceeds threshold",
                    location="Field: amount",
                    remediation="Verify large transaction amount",
                    timestamp=datetime.now()
                ))

        return violations

    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data"""
        if isinstance(data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
            sanitized = data
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            return sanitized.strip()

        elif isinstance(data, dict):
            return {key: self.sanitize_input(value) for key, value in data.items()}

        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]

        return data


class AccessControlGuard:
    """Access control and authorization guardrails"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failed_attempts = {}  # In production, use Redis or database

    def validate_access(self, user_id: str, resource: str, action: str) -> Tuple[bool, Optional[SecurityViolation]]:
        """Validate user access to resource"""

        # Check rate limiting
        if self._is_rate_limited(user_id):
            return False, SecurityViolation(
                violation_type="rate_limit_exceeded",
                severity=SecurityLevel.HIGH,
                description=f"Rate limit exceeded for user {user_id}",
                location=f"User: {user_id}",
                remediation="Wait before retrying",
                timestamp=datetime.now()
            )

        # Check resource permissions (simplified)
        if not self._has_permission(user_id, resource, action):
            self._record_failed_attempt(user_id)
            return False, SecurityViolation(
                violation_type="unauthorized_access",
                severity=SecurityLevel.HIGH,
                description=f"User {user_id} lacks permission for {action} on {resource}",
                location=f"Resource: {resource}",
                remediation="Request proper authorization",
                timestamp=datetime.now()
            )

        return True, None

    def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=5)

        if user_id in self.failed_attempts:
            recent_attempts = [
                attempt for attempt in self.failed_attempts[user_id]
                if attempt > window_start
            ]
            self.failed_attempts[user_id] = recent_attempts
            return len(recent_attempts) >= 10  # 10 attempts per 5 minutes

        return False

    def _has_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has permission (simplified implementation)"""
        # In a real implementation, this would check against a database or IAM system

        # Admin users have all permissions
        if user_id.startswith('admin_'):
            return True

        # Auditor permissions
        if user_id.startswith('auditor_'):
            allowed_actions = ['read', 'audit', 'analyze']
            return action in allowed_actions

        # Regular user permissions
        if action == 'read' and resource in ['dashboard', 'reports']:
            return True

        return False

    def _record_failed_attempt(self, user_id: str):
        """Record a failed access attempt"""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []

        self.failed_attempts[user_id].append(datetime.now())


class CryptographicGuard:
    """Cryptographic operations guardrails"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)

    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash sensitive data with salt"""
        if salt is None:
            salt = secrets.token_hex(16)

        # Use PBKDF2 for password-like data
        hashed = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hashed.hex(), salt

    def verify_hash(self, data: str, hashed: str, salt: str) -> bool:
        """Verify hashed data"""
        computed_hash, _ = self.hash_sensitive_data(data, salt)
        return secrets.compare_digest(computed_hash, hashed)

    def validate_encryption_requirements(self, data_type: str, is_encrypted: bool) -> Optional[SecurityViolation]:
        """Validate encryption requirements for data types"""
        encryption_required = ['ssn', 'credit_card', 'bank_account', 'api_key']

        if data_type in encryption_required and not is_encrypted:
            return SecurityViolation(
                violation_type="encryption_required",
                severity=SecurityLevel.CRITICAL,
                description=f"Data type {data_type} must be encrypted",
                location=f"Data type: {data_type}",
                remediation="Encrypt sensitive data before storage",
                timestamp=datetime.now()
            )

        return None


class SecurityGuardRailsManager:
    """Main security guardrails manager"""

    def __init__(self):
        self.data_guard = DataClassificationGuard()
        self.input_guard = InputValidationGuard()
        self.access_guard = AccessControlGuard()
        self.crypto_guard = CryptographicGuard()
        self.logger = logging.getLogger(__name__)

    def comprehensive_security_check(
        self,
        data: Dict[str, Any],
        user_id: str,
        resource: str,
        action: str
    ) -> Tuple[bool, List[SecurityViolation]]:
        """Perform comprehensive security check"""

        all_violations = []

        # 1. Access control check
        access_allowed, access_violation = self.access_guard.validate_access(user_id, resource, action)
        if not access_allowed and access_violation:
            all_violations.append(access_violation)
            return False, all_violations

        # 2. Input validation
        input_violations = self.input_guard.validate_financial_data(data)
        all_violations.extend(input_violations)

        # 3. Sensitive data detection
        text_data = str(data)
        data_violations = self.data_guard.scan_text(text_data, f"{resource}:{action}")
        all_violations.extend(data_violations)

        # 4. Check if any critical violations exist
        critical_violations = [v for v in all_violations if v.severity == SecurityLevel.CRITICAL]
        if critical_violations:
            return False, all_violations

        # 5. Log security events
        self._log_security_event(user_id, resource, action, all_violations)

        return True, all_violations

    def sanitize_and_mask_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and mask sensitive data"""
        # First sanitize
        sanitized = self.input_guard.sanitize_input(data)

        # Then mask sensitive data in string fields
        if isinstance(sanitized, dict):
            for key, value in sanitized.items():
                if isinstance(value, str):
                    sanitized[key] = self.data_guard.mask_sensitive_data(value)

        return sanitized

    def _log_security_event(self, user_id: str, resource: str, action: str, violations: List[SecurityViolation]):
        """Log security events for monitoring"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'violations_count': len(violations),
            'violation_types': [v.violation_type for v in violations],
            'max_severity': max([v.severity.value for v in violations]) if violations else 'none'
        }

        if violations:
            self.logger.warning(f"Security violations detected: {event}")
        else:
            self.logger.info(f"Security check passed: {event}")


# Global security manager instance
security_manager = SecurityGuardRailsManager()