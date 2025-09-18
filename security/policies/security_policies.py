"""
Security Policies for Financial Audit System
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json


class PolicyType(Enum):
    DATA_CLASSIFICATION = "data_classification"
    ACCESS_CONTROL = "access_control"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"
    COMPLIANCE = "compliance"


class ComplianceFramework(Enum):
    SOX = "sarbanes_oxley"
    GAAP = "gaap"
    IFRS = "ifrs"
    GDPR = "gdpr"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


@dataclass
class SecurityPolicy:
    """Represents a security policy"""
    id: str
    name: str
    policy_type: PolicyType
    compliance_frameworks: List[ComplianceFramework]
    description: str
    requirements: List[str]
    enforcement_level: str  # "advisory", "warning", "blocking"
    effective_date: datetime
    review_date: datetime
    owner: str


class DataClassificationPolicy:
    """Data classification security policies"""

    @staticmethod
    def get_policies() -> List[SecurityPolicy]:
        return [
            SecurityPolicy(
                id="DC-001",
                name="Financial Data Classification",
                policy_type=PolicyType.DATA_CLASSIFICATION,
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.GAAP],
                description="Classification requirements for financial data",
                requirements=[
                    "All financial transaction data must be classified as CONFIDENTIAL or higher",
                    "Personal identifiable information (PII) must be classified as RESTRICTED",
                    "Audit reports must be classified as CONFIDENTIAL",
                    "Risk assessments must be classified as INTERNAL USE",
                    "Publicly available financial statements may be classified as PUBLIC"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="CISO"
            ),
            SecurityPolicy(
                id="DC-002",
                name="Sensitive Data Handling",
                policy_type=PolicyType.DATA_CLASSIFICATION,
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.GDPR],
                description="Handling requirements for sensitive data",
                requirements=[
                    "SSNs, credit card numbers, and bank accounts must be encrypted at rest",
                    "Sensitive data must be masked in logs and non-production environments",
                    "Data retention periods must be enforced based on classification",
                    "Cross-border data transfers must comply with applicable regulations"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="Data Protection Officer"
            )
        ]


class AccessControlPolicy:
    """Access control security policies"""

    @staticmethod
    def get_policies() -> List[SecurityPolicy]:
        return [
            SecurityPolicy(
                id="AC-001",
                name="Role-Based Access Control",
                policy_type=PolicyType.ACCESS_CONTROL,
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.SOC2],
                description="Role-based access control requirements",
                requirements=[
                    "Users must be assigned roles with minimum necessary privileges",
                    "Segregation of duties must be enforced for critical functions",
                    "Access reviews must be conducted quarterly",
                    "Emergency access must be logged and reviewed within 24 hours",
                    "Privileged access requires dual approval"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 6, 30),
                owner="Security Team"
            ),
            SecurityPolicy(
                id="AC-002",
                name="Multi-Factor Authentication",
                policy_type=PolicyType.ACCESS_CONTROL,
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.SOC2],
                description="Multi-factor authentication requirements",
                requirements=[
                    "MFA required for all privileged accounts",
                    "MFA required for remote access",
                    "MFA required for financial system access",
                    "Hardware tokens required for administrator accounts",
                    "MFA bypass requires approval and logging"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="Identity Management Team"
            ),
            SecurityPolicy(
                id="AC-003",
                name="Session Management",
                policy_type=PolicyType.ACCESS_CONTROL,
                compliance_frameworks=[ComplianceFramework.SOX],
                description="Session management requirements",
                requirements=[
                    "Sessions must timeout after 30 minutes of inactivity",
                    "Concurrent sessions limited to 3 per user",
                    "Session tokens must be cryptographically secure",
                    "Session data must not be stored in client-side storage",
                    "Failed login attempts trigger account lockout after 5 attempts"
                ],
                enforcement_level="warning",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="Application Security Team"
            )
        ]


class EncryptionPolicy:
    """Encryption security policies"""

    @staticmethod
    def get_policies() -> List[SecurityPolicy]:
        return [
            SecurityPolicy(
                id="EN-001",
                name="Data Encryption Standards",
                policy_type=PolicyType.ENCRYPTION,
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.GDPR],
                description="Encryption standards for data protection",
                requirements=[
                    "AES-256 encryption required for data at rest",
                    "TLS 1.3 required for data in transit",
                    "Key management must use FIPS 140-2 Level 3 or higher",
                    "Encryption keys must be rotated annually",
                    "Database encryption must use transparent data encryption (TDE)"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="Cryptography Team"
            ),
            SecurityPolicy(
                id="EN-002",
                name="Key Management",
                policy_type=PolicyType.ENCRYPTION,
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.ISO27001],
                description="Cryptographic key management requirements",
                requirements=[
                    "Keys must be generated using approved random number generators",
                    "Key storage must be in hardware security modules (HSM)",
                    "Key access must be logged and monitored",
                    "Master keys must be escrowed with dual control",
                    "Compromised keys must be revoked within 4 hours"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 6, 30),
                owner="Cryptography Team"
            )
        ]


class AuditLoggingPolicy:
    """Audit logging security policies"""

    @staticmethod
    def get_policies() -> List[SecurityPolicy]:
        return [
            SecurityPolicy(
                id="AL-001",
                name="Comprehensive Audit Logging",
                policy_type=PolicyType.AUDIT_LOGGING,
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.SOC2],
                description="Comprehensive audit logging requirements",
                requirements=[
                    "All user authentication events must be logged",
                    "All data access and modifications must be logged",
                    "All administrative actions must be logged",
                    "All system errors and security events must be logged",
                    "Logs must include timestamp, user ID, action, and outcome"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="Security Operations Team"
            ),
            SecurityPolicy(
                id="AL-002",
                name="Log Retention and Protection",
                policy_type=PolicyType.AUDIT_LOGGING,
                compliance_frameworks=[ComplianceFramework.SOX],
                description="Log retention and protection requirements",
                requirements=[
                    "Security logs must be retained for 7 years",
                    "Financial audit logs must be retained for 7 years",
                    "System logs must be retained for 1 year",
                    "Logs must be protected against tampering",
                    "Log integrity must be verified using digital signatures"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="Records Management Team"
            )
        ]


class CompliancePolicy:
    """Compliance-specific security policies"""

    @staticmethod
    def get_policies() -> List[SecurityPolicy]:
        return [
            SecurityPolicy(
                id="CP-001",
                name="SOX IT General Controls",
                policy_type=PolicyType.COMPLIANCE,
                compliance_frameworks=[ComplianceFramework.SOX],
                description="Sarbanes-Oxley IT General Controls requirements",
                requirements=[
                    "Change management controls for financial applications",
                    "User access controls for financial systems",
                    "Data backup and recovery procedures",
                    "Business continuity and disaster recovery plans",
                    "Security monitoring and incident response procedures"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="Compliance Team"
            ),
            SecurityPolicy(
                id="CP-002",
                name="Financial Reporting Controls",
                policy_type=PolicyType.COMPLIANCE,
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.GAAP],
                description="Financial reporting control requirements",
                requirements=[
                    "Automated controls for financial calculations",
                    "Approval workflows for financial adjustments",
                    "Reconciliation procedures for financial data",
                    "Period-end close procedures and controls",
                    "Management review and certification processes"
                ],
                enforcement_level="blocking",
                effective_date=datetime(2024, 1, 1),
                review_date=datetime(2024, 12, 31),
                owner="Financial Controller"
            )
        ]


class SecurityPolicyManager:
    """Manager for security policies"""

    def __init__(self):
        self.policies = self._load_all_policies()

    def _load_all_policies(self) -> List[SecurityPolicy]:
        """Load all security policies"""
        all_policies = []

        all_policies.extend(DataClassificationPolicy.get_policies())
        all_policies.extend(AccessControlPolicy.get_policies())
        all_policies.extend(EncryptionPolicy.get_policies())
        all_policies.extend(AuditLoggingPolicy.get_policies())
        all_policies.extend(CompliancePolicy.get_policies())

        return all_policies

    def get_policies_by_type(self, policy_type: PolicyType) -> List[SecurityPolicy]:
        """Get policies by type"""
        return [p for p in self.policies if p.policy_type == policy_type]

    def get_policies_by_framework(self, framework: ComplianceFramework) -> List[SecurityPolicy]:
        """Get policies by compliance framework"""
        return [p for p in self.policies if framework in p.compliance_frameworks]

    def get_policy_by_id(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get policy by ID"""
        for policy in self.policies:
            if policy.id == policy_id:
                return policy
        return None

    def get_enforcement_blocking_policies(self) -> List[SecurityPolicy]:
        """Get policies with blocking enforcement"""
        return [p for p in self.policies if p.enforcement_level == "blocking"]

    def check_policy_compliance(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Check compliance against applicable policies"""
        violations = {}

        for policy in self.policies:
            policy_violations = self._check_policy(policy, context)
            if policy_violations:
                violations[policy.id] = policy_violations

        return violations

    def _check_policy(self, policy: SecurityPolicy, context: Dict[str, Any]) -> List[str]:
        """Check a specific policy against context"""
        violations = []

        # This is a simplified implementation
        # In practice, each policy would have specific validation logic

        if policy.policy_type == PolicyType.DATA_CLASSIFICATION:
            if not context.get('data_classified', False):
                violations.append("Data must be classified according to policy")

        elif policy.policy_type == PolicyType.ACCESS_CONTROL:
            if not context.get('user_authenticated', False):
                violations.append("User must be authenticated")
            if not context.get('mfa_verified', False) and policy.id == "AC-002":
                violations.append("Multi-factor authentication required")

        elif policy.policy_type == PolicyType.ENCRYPTION:
            if not context.get('data_encrypted', False):
                violations.append("Data must be encrypted according to policy")

        elif policy.policy_type == PolicyType.AUDIT_LOGGING:
            if not context.get('audit_logged', False):
                violations.append("Action must be audit logged")

        return violations

    def export_policies(self, format: str = "json") -> str:
        """Export policies to specified format"""
        if format == "json":
            policy_data = []
            for policy in self.policies:
                policy_data.append({
                    'id': policy.id,
                    'name': policy.name,
                    'type': policy.policy_type.value,
                    'frameworks': [f.value for f in policy.compliance_frameworks],
                    'description': policy.description,
                    'requirements': policy.requirements,
                    'enforcement_level': policy.enforcement_level,
                    'effective_date': policy.effective_date.isoformat(),
                    'review_date': policy.review_date.isoformat(),
                    'owner': policy.owner
                })
            return json.dumps(policy_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global policy manager instance
policy_manager = SecurityPolicyManager()