"""
Redis Configuration for Financial Audit System
"""

import redis
from typing import Optional, Dict, Any
import json
import logging
from backend.core.config import settings


class RedisManager:
    """Redis connection and operations manager"""

    def __init__(self, url: Optional[str] = None):
        self.url = url or settings.REDIS_URL
        self.client = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> redis.Redis:
        """Establish Redis connection"""
        try:
            self.client = redis.from_url(
                self.url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )

            # Test connection
            self.client.ping()
            self.logger.info("Redis connection established successfully")
            return self.client

        except redis.RedisError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get_client(self) -> redis.Redis:
        """Get Redis client instance"""
        if self.client is None:
            self.connect()
        return self.client

    def set_json(self, key: str, value: Dict[str, Any], ex: Optional[int] = None) -> bool:
        """Store JSON data in Redis"""
        try:
            client = self.get_client()
            json_str = json.dumps(value)
            return client.set(key, json_str, ex=ex)
        except Exception as e:
            self.logger.error(f"Error setting JSON key {key}: {e}")
            return False

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve JSON data from Redis"""
        try:
            client = self.get_client()
            value = client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            self.logger.error(f"Error getting JSON key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        try:
            client = self.get_client()
            return bool(client.delete(key))
        except Exception as e:
            self.logger.error(f"Error deleting key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            client = self.get_client()
            return bool(client.exists(key))
        except Exception as e:
            self.logger.error(f"Error checking key existence {key}: {e}")
            return False

    def set_ttl(self, key: str, seconds: int) -> bool:
        """Set TTL for existing key"""
        try:
            client = self.get_client()
            return bool(client.expire(key, seconds))
        except Exception as e:
            self.logger.error(f"Error setting TTL for key {key}: {e}")
            return False

    def get_keys_pattern(self, pattern: str) -> list:
        """Get keys matching pattern"""
        try:
            client = self.get_client()
            return client.keys(pattern)
        except Exception as e:
            self.logger.error(f"Error getting keys with pattern {pattern}: {e}")
            return []


class AuditCacheManager:
    """Cache manager for audit-specific operations"""

    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.audit_prefix = "audit:"
        self.session_prefix = "session:"
        self.result_prefix = "result:"

    def cache_audit_session(self, audit_id: str, session_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache audit session data"""
        key = f"{self.session_prefix}{audit_id}"
        return self.redis.set_json(key, session_data, ex=ttl)

    def get_audit_session(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve audit session data"""
        key = f"{self.session_prefix}{audit_id}"
        return self.redis.get_json(key)

    def cache_audit_results(self, audit_id: str, results: Dict[str, Any], ttl: int = 86400) -> bool:
        """Cache audit results (24-hour default TTL)"""
        key = f"{self.result_prefix}{audit_id}"
        return self.redis.set_json(key, results, ex=ttl)

    def get_audit_results(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached audit results"""
        key = f"{self.result_prefix}{audit_id}"
        return self.redis.get_json(key)

    def cache_risk_analysis(self, company_id: str, risk_data: Dict[str, Any], ttl: int = 7200) -> bool:
        """Cache risk analysis results (2-hour default TTL)"""
        key = f"risk:{company_id}"
        return self.redis.set_json(key, risk_data, ex=ttl)

    def get_risk_analysis(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached risk analysis"""
        key = f"risk:{company_id}"
        return self.redis.get_json(key)

    def invalidate_audit_cache(self, audit_id: str) -> bool:
        """Invalidate all cache entries for an audit"""
        patterns = [
            f"{self.session_prefix}{audit_id}",
            f"{self.result_prefix}{audit_id}",
            f"{self.audit_prefix}{audit_id}:*"
        ]

        success = True
        for pattern in patterns:
            keys = self.redis.get_keys_pattern(pattern)
            for key in keys:
                success &= self.redis.delete(key)

        return success

    def get_active_audits(self) -> list:
        """Get list of active audit sessions"""
        pattern = f"{self.session_prefix}*"
        keys = self.redis.get_keys_pattern(pattern)
        return [key.replace(self.session_prefix, "") for key in keys]


# Global Redis manager instance
redis_manager = RedisManager()
audit_cache = AuditCacheManager(redis_manager)