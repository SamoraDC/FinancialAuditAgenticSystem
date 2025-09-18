"""
Performance testing suite for Financial Audit Agentic System
Using Locust for load testing and performance validation
"""

from locust import HttpUser, task, between
import json
import random
from datetime import datetime, timedelta


class AuditSystemUser(HttpUser):
    """Simulates a user interacting with the Financial Audit System"""

    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks

    def on_start(self):
        """Setup user session"""
        self.auth_token = None
        self.audit_id = None
        self.login()

    def login(self):
        """Authenticate user"""
        response = self.client.post("/auth/login", json={
            "username": f"test_user_{random.randint(1, 1000)}",
            "password": "test_password"
        })

        if response.status_code == 200:
            self.auth_token = response.json().get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })

    @task(5)
    def health_check(self):
        """Check system health"""
        self.client.get("/health")

    @task(3)
    def upload_document(self):
        """Upload a financial document"""
        # Simulate file upload
        files = {
            'file': ('test_invoice.pdf', b'fake_pdf_content', 'application/pdf')
        }

        response = self.client.post(
            "/api/v1/documents/upload",
            files=files,
            headers={"Authorization": f"Bearer {self.auth_token}"}
        )

        if response.status_code == 201:
            self.audit_id = response.json().get("audit_id")

    @task(4)
    def start_audit(self):
        """Start an audit process"""
        if not self.audit_id:
            return

        response = self.client.post(
            f"/api/v1/audits/{self.audit_id}/start",
            json={
                "audit_type": "comprehensive",
                "compliance_frameworks": ["SOX", "GAAP"],
                "priority": "medium"
            }
        )

    @task(6)
    def check_audit_status(self):
        """Check audit progress"""
        if not self.audit_id:
            return

        self.client.get(f"/api/v1/audits/{self.audit_id}/status")

    @task(2)
    def get_audit_results(self):
        """Retrieve audit results"""
        if not self.audit_id:
            return

        self.client.get(f"/api/v1/audits/{self.audit_id}/results")

    @task(2)
    def anomaly_detection(self):
        """Test anomaly detection endpoint"""
        sample_transactions = [
            {
                "amount": random.uniform(100, 10000),
                "vendor": f"Vendor_{random.randint(1, 100)}",
                "date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                "category": random.choice(["office_supplies", "travel", "software", "consulting"])
            }
            for _ in range(random.randint(5, 20))
        ]

        self.client.post(
            "/api/v1/analysis/anomaly-detection",
            json={"transactions": sample_transactions}
        )

    @task(1)
    def get_dashboard_data(self):
        """Load dashboard data"""
        self.client.get("/api/v1/dashboard/overview")

    @task(1)
    def export_report(self):
        """Export audit report"""
        if not self.audit_id:
            return

        self.client.post(
            f"/api/v1/audits/{self.audit_id}/export",
            json={"format": "pdf", "include_details": True}
        )


class AdminUser(HttpUser):
    """Simulates an admin user with elevated privileges"""

    wait_time = between(2, 8)
    weight = 1  # Lower weight for admin users

    def on_start(self):
        """Setup admin session"""
        response = self.client.post("/auth/login", json={
            "username": "admin_user",
            "password": "admin_password"
        })

        if response.status_code == 200:
            self.auth_token = response.json().get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })

    @task(3)
    def get_system_metrics(self):
        """Check system performance metrics"""
        self.client.get("/api/v1/admin/metrics")

    @task(2)
    def manage_users(self):
        """User management operations"""
        self.client.get("/api/v1/admin/users")

    @task(1)
    def system_configuration(self):
        """Access system configuration"""
        self.client.get("/api/v1/admin/config")


class MLModelUser(HttpUser):
    """Simulates ML model training and inference workloads"""

    wait_time = between(5, 15)
    weight = 1  # Fewer ML operations

    @task(2)
    def train_model(self):
        """Trigger model training"""
        self.client.post("/api/v1/ml/train", json={
            "model_type": "anomaly_detection",
            "training_config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        })

    @task(3)
    def model_inference(self):
        """Run model inference"""
        sample_data = {
            "features": [random.uniform(0, 1) for _ in range(20)]
        }

        self.client.post("/api/v1/ml/predict", json=sample_data)

    @task(1)
    def model_status(self):
        """Check model status"""
        self.client.get("/api/v1/ml/models/status")


# Performance test scenarios
class QuickTest(AuditSystemUser):
    """Quick performance test - high frequency, simple operations"""
    wait_time = between(0.5, 2)


class StressTest(AuditSystemUser):
    """Stress test - sustained high load"""
    wait_time = between(0.1, 1)


class SoakTest(AuditSystemUser):
    """Soak test - long duration, moderate load"""
    wait_time = between(2, 10)