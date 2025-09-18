"""
LangGraph Redis Checkpoint Configuration
Optimized for Financial Audit Agentic System
"""

import redis
from langgraph.checkpoint.redis import RedisSaver
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FinancialAuditCheckpointer:
    """
    Custom Redis checkpoint manager for Financial Audit System
    Provides enhanced persistence and auditability features
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        namespace: str = "financial_audit",
        ttl_days: int = 2555,  # 7 years for SOX compliance
        **redis_kwargs
    ):
        self.redis_url = redis_url
        self.namespace = namespace
        self.ttl_seconds = ttl_days * 24 * 60 * 60

        # Initialize Redis connection
        self.redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            **redis_kwargs
        )

        # Initialize LangGraph Redis saver
        self.saver = RedisSaver(
            redis_client=self.redis_client,
            namespace=namespace
        )

        logger.info(f"Initialized FinancialAuditCheckpointer with namespace: {namespace}")

    def get_checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        """Generate standardized checkpoint key"""
        return f"{self.namespace}:checkpoint:{thread_id}:{checkpoint_id}"

    def get_audit_trail_key(self, thread_id: str) -> str:
        """Generate audit trail key for compliance tracking"""
        return f"{self.namespace}:audit_trail:{thread_id}"

    def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save checkpoint with enhanced audit trail
        """
        try:
            # Save using LangGraph's built-in saver
            self.saver.put(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint=state,
                metadata=metadata or {}
            )

            # Create audit trail entry
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "action": "checkpoint_saved",
                "metadata": metadata or {},
                "state_keys": list(state.keys()) if state else [],
                "compliance_flags": self._extract_compliance_flags(state)
            }

            # Store audit trail with TTL for compliance
            audit_key = self.get_audit_trail_key(thread_id)
            self.redis_client.lpush(audit_key, json.dumps(audit_entry))
            self.redis_client.expire(audit_key, self.ttl_seconds)

            logger.info(f"Checkpoint saved: {thread_id}:{checkpoint_id}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint {thread_id}:{checkpoint_id}: {e}")
            raise

    def load_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint with audit logging
        """
        try:
            # Load using LangGraph's built-in saver
            checkpoint = self.saver.get(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id
            )

            if checkpoint:
                # Log checkpoint access for audit trail
                audit_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "action": "checkpoint_loaded"
                }

                audit_key = self.get_audit_trail_key(thread_id)
                self.redis_client.lpush(audit_key, json.dumps(audit_entry))

                logger.info(f"Checkpoint loaded: {thread_id}:{checkpoint_id}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {thread_id}:{checkpoint_id}: {e}")
            raise

    def list_checkpoints(self, thread_id: str) -> list:
        """List all checkpoints for a thread"""
        try:
            return self.saver.list(thread_id=thread_id)
        except Exception as e:
            logger.error(f"Failed to list checkpoints for {thread_id}: {e}")
            raise

    def get_audit_trail(self, thread_id: str, limit: int = 100) -> list:
        """
        Retrieve audit trail for compliance reporting
        """
        try:
            audit_key = self.get_audit_trail_key(thread_id)
            entries = self.redis_client.lrange(audit_key, 0, limit - 1)
            return [json.loads(entry) for entry in entries]
        except Exception as e:
            logger.error(f"Failed to get audit trail for {thread_id}: {e}")
            raise

    def cleanup_expired_checkpoints(self, days_old: int = 30) -> int:
        """
        Cleanup old checkpoints while preserving compliance data
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            pattern = f"{self.namespace}:checkpoint:*"

            keys = self.redis_client.keys(pattern)
            deleted_count = 0

            for key in keys:
                # Check if checkpoint is older than cutoff
                # Note: This is a simplified version - implement proper date checking
                ttl = self.redis_client.ttl(key)
                if ttl > 0 and ttl < (self.ttl_seconds - days_old * 24 * 60 * 60):
                    continue  # Keep for compliance

                # Only delete non-compliance-critical checkpoints
                if self._is_compliance_critical(key):
                    continue

                self.redis_client.delete(key)
                deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} expired checkpoints")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired checkpoints: {e}")
            raise

    def _extract_compliance_flags(self, state: Dict[str, Any]) -> list:
        """Extract compliance-relevant flags from state"""
        flags = []

        if state:
            # Check for SOX-related data
            if any(key.lower().startswith('sox') for key in state.keys()):
                flags.append('SOX')

            # Check for financial data
            if any(key.lower() in ['financial_data', 'transactions', 'amounts']
                   for key in state.keys()):
                flags.append('FINANCIAL')

            # Check for audit findings
            if 'findings' in state or 'anomalies' in state:
                flags.append('AUDIT_FINDINGS')

        return flags

    def _is_compliance_critical(self, key: str) -> bool:
        """Determine if a checkpoint is compliance-critical"""
        # This is a simplified check - implement proper compliance rules
        return 'sox' in key.lower() or 'financial' in key.lower()

    def health_check(self) -> Dict[str, Any]:
        """Check Redis connection and checkpoint system health"""
        try:
            # Test Redis connection
            self.redis_client.ping()

            # Get Redis info
            redis_info = self.redis_client.info()

            # Test checkpoint operations
            test_thread = "health_check_test"
            test_state = {"test": True, "timestamp": datetime.utcnow().isoformat()}

            self.save_checkpoint(test_thread, "test", test_state)
            loaded_state = self.load_checkpoint(test_thread, "test")

            # Cleanup test data
            self.redis_client.delete(f"{self.namespace}:checkpoint:{test_thread}:test")
            self.redis_client.delete(self.get_audit_trail_key(test_thread))

            return {
                "status": "healthy",
                "redis_connected": True,
                "checkpoint_operations": "working",
                "redis_memory_usage": redis_info.get('used_memory_human'),
                "redis_version": redis_info.get('redis_version'),
                "namespace": self.namespace,
                "ttl_days": self.ttl_seconds // (24 * 60 * 60)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "redis_connected": False
            }


# Factory function for easy initialization
def create_checkpoint_manager(
    redis_url: str = "redis://localhost:6379/0",
    namespace: str = "financial_audit",
    **kwargs
) -> FinancialAuditCheckpointer:
    """
    Factory function to create a configured checkpoint manager
    """
    return FinancialAuditCheckpointer(
        redis_url=redis_url,
        namespace=namespace,
        **kwargs
    )