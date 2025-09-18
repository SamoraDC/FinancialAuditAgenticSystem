"""
Locust performance test file for Financial Audit System
"""

from locust import HttpUser, task, between
import json


class AuditSystemUser(HttpUser):
    """Simulated user for load testing the audit system"""

    wait_time = between(1, 3)

    def on_start(self):
        """Setup for each user"""
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    @task(3)
    def health_check(self):
        """Test health endpoint - most frequent operation"""
        self.client.get("/health")

    @task(2)
    def api_root(self):
        """Test API root endpoint"""
        self.client.get("/api/v1", headers=self.headers)

    @task(1)
    def test_audit_endpoints(self):
        """Test audit-related endpoints if available"""
        # Test audit sessions endpoint
        response = self.client.get("/api/v1/audit/sessions", headers=self.headers)

        if response.status_code == 200:
            # If successful, try to get session details
            sessions = response.json()
            if isinstance(sessions, list) and len(sessions) > 0:
                session_id = sessions[0].get('id')
                if session_id:
                    self.client.get(f"/api/v1/audit/sessions/{session_id}", headers=self.headers)


class HighLoadUser(HttpUser):
    """High-load user simulation for stress testing"""

    wait_time = between(0.1, 0.5)

    @task
    def rapid_health_checks(self):
        """Rapid health checks for stress testing"""
        self.client.get("/health")


# Configuration for different test scenarios
class StressTestUser(HttpUser):
    """Stress test user with concurrent operations"""

    wait_time = between(0, 1)

    @task(5)
    def health_stress(self):
        """Stress test health endpoint"""
        self.client.get("/health")

    @task(1)
    def api_stress(self):
        """Stress test API endpoints"""
        self.client.get("/api/v1", headers={'Accept': 'application/json'})