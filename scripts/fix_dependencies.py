#!/usr/bin/env python3
"""
🔧 Automatic Dependency Fix Script
==================================

This script automatically fixes common dependency issues found during CI/CD validation.
It provides intelligent auto-repair for the most common dependency failures.

Features:
- Auto-fixes version conflicts
- Resolves missing dependencies
- Updates lock files safely
- Creates backup configurations
- Provides rollback capabilities
"""

import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import toml
from datetime import datetime


class DependencyFixer:
    """Intelligent dependency issue resolver"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.requirements_path = self.project_root / "requirements.txt"
        self.backup_dir = self.project_root / ".dependency_backups"
        self.fixes_applied = []

    def create_backup(self) -> str:
        """Create backup of current dependency files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        print(f"💾 Creating backup at {backup_path}")

        # Backup pyproject.toml
        if self.pyproject_path.exists():
            shutil.copy2(self.pyproject_path, backup_path / "pyproject.toml")

        # Backup requirements.txt
        if self.requirements_path.exists():
            shutil.copy2(self.requirements_path, backup_path / "requirements.txt")

        # Backup uv.lock if exists
        uv_lock = self.project_root / "uv.lock"
        if uv_lock.exists():
            shutil.copy2(uv_lock, backup_path / "uv.lock")

        print(f"✅ Backup created successfully")
        return str(backup_path)

    def fix_python_version_mismatch(self) -> bool:
        """Fix Python version mismatches between files"""
        print("🔧 Fixing Python version mismatches...")

        try:
            # Read .python-version if it exists
            python_version_file = self.project_root / ".python-version"
            if python_version_file.exists():
                target_version = python_version_file.read_text().strip()
                print(f"🎯 Target Python version from .python-version: {target_version}")

                # Update pyproject.toml
                if self.pyproject_path.exists():
                    with open(self.pyproject_path, 'r') as f:
                        config = toml.load(f)

                    # Update requires-python
                    if 'project' not in config:
                        config['project'] = {}

                    old_requires = config['project'].get('requires-python', '')
                    config['project']['requires-python'] = f">={target_version}"

                    with open(self.pyproject_path, 'w') as f:
                        toml.dump(config, f)

                    print(f"✅ Updated requires-python: {old_requires} → >={target_version}")
                    self.fixes_applied.append(f"Updated Python version requirement to >={target_version}")
                    return True
            else:
                print("⚠️ No .python-version file found")
                return True

        except Exception as e:
            print(f"❌ Failed to fix Python version mismatch: {e}")
            return False

    def consolidate_dependencies(self) -> bool:
        """Consolidate dependencies from requirements.txt into pyproject.toml"""
        print("🔧 Consolidating dependencies...")

        if not self.requirements_path.exists():
            print("✅ No requirements.txt to consolidate")
            return True

        try:
            # Read current pyproject.toml
            if not self.pyproject_path.exists():
                print("❌ pyproject.toml not found")
                return False

            with open(self.pyproject_path, 'r') as f:
                config = toml.load(f)

            # Read requirements.txt
            with open(self.requirements_path, 'r') as f:
                req_lines = f.readlines()

            # Parse requirements
            requirements = []
            dev_requirements = []

            for line in req_lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Categorize as dev or prod dependency
                    if any(dev_keyword in line.lower() for dev_keyword in ['test', 'pytest', 'bandit', 'safety', 'lint', 'format', 'black', 'isort', 'mypy', 'flake8', 'coverage']):
                        dev_requirements.append(line)
                    else:
                        requirements.append(line)

            # Get existing dependencies
            existing_deps = set(config.get('project', {}).get('dependencies', []))
            existing_dev_deps = set(config.get('project', {}).get('optional-dependencies', {}).get('dev', []))

            # Add new requirements (avoiding duplicates)
            new_deps = []
            for req in requirements:
                pkg_name = req.split('>=')[0].split('==')[0].split('[')[0].strip()
                if not any(pkg_name in dep for dep in existing_deps):
                    new_deps.append(req)

            new_dev_deps = []
            for req in dev_requirements:
                pkg_name = req.split('>=')[0].split('==')[0].split('[')[0].strip()
                if not any(pkg_name in dep for dep in existing_dev_deps):
                    new_dev_deps.append(req)

            # Update config
            if 'project' not in config:
                config['project'] = {}

            if new_deps:
                current_deps = config['project'].get('dependencies', [])
                config['project']['dependencies'] = current_deps + new_deps
                print(f"✅ Added {len(new_deps)} production dependencies")

            if new_dev_deps:
                if 'optional-dependencies' not in config['project']:
                    config['project']['optional-dependencies'] = {}
                if 'dev' not in config['project']['optional-dependencies']:
                    config['project']['optional-dependencies']['dev'] = []

                current_dev_deps = config['project']['optional-dependencies']['dev']
                config['project']['optional-dependencies']['dev'] = current_dev_deps + new_dev_deps
                print(f"✅ Added {len(new_dev_deps)} development dependencies")

            # Write updated config
            with open(self.pyproject_path, 'w') as f:
                toml.dump(config, f)

            # Create backup of requirements.txt and rename it
            backup_req = self.requirements_path.with_suffix('.txt.backup')
            shutil.copy2(self.requirements_path, backup_req)
            print(f"✅ Backed up requirements.txt to {backup_req}")

            self.fixes_applied.append(f"Consolidated {len(new_deps + new_dev_deps)} dependencies from requirements.txt")
            return True

        except Exception as e:
            print(f"❌ Failed to consolidate dependencies: {e}")
            return False

    def fix_version_conflicts(self) -> bool:
        """Fix common version conflicts"""
        print("🔧 Fixing version conflicts...")

        try:
            # Common conflict resolution patterns
            conflict_fixes = {
                # Example: if both pydantic v1 and v2 are present, prefer v2
                'pydantic': '>=2.0.0',
                'fastapi': '>=0.100.0',
                'uvicorn': '>=0.20.0',
                'langchain': '>=0.1.0',
                'pytest': '>=7.0.0',
            }

            with open(self.pyproject_path, 'r') as f:
                config = toml.load(f)

            dependencies = config.get('project', {}).get('dependencies', [])
            updated_deps = []
            fixes_made = 0

            for dep in dependencies:
                if isinstance(dep, str):
                    pkg_name = dep.split('>=')[0].split('==')[0].split('[')[0].strip()
                    if pkg_name in conflict_fixes:
                        # Apply the fix
                        new_dep = f"{pkg_name}{conflict_fixes[pkg_name]}"
                        updated_deps.append(new_dep)
                        if new_dep != dep:
                            print(f"🔄 Updated {pkg_name}: {dep} → {new_dep}")
                            fixes_made += 1
                    else:
                        updated_deps.append(dep)
                else:
                    updated_deps.append(dep)

            if fixes_made > 0:
                config['project']['dependencies'] = updated_deps
                with open(self.pyproject_path, 'w') as f:
                    toml.dump(config, f)
                self.fixes_applied.append(f"Fixed {fixes_made} version conflicts")

            print(f"✅ Applied {fixes_made} version conflict fixes")
            return True

        except Exception as e:
            print(f"❌ Failed to fix version conflicts: {e}")
            return False

    def update_lock_file(self) -> bool:
        """Update and validate lock file"""
        print("🔧 Updating lock file...")

        try:
            # Try to update uv.lock
            result = subprocess.run(
                ['uv', 'lock', '--upgrade'],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print("✅ Lock file updated successfully")
                self.fixes_applied.append("Updated and validated lock file")
                return True
            else:
                print(f"⚠️ Lock file update had warnings:\n{result.stderr}")
                # Try without upgrade
                result2 = subprocess.run(
                    ['uv', 'lock'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result2.returncode == 0:
                    print("✅ Lock file regenerated successfully")
                    return True
                else:
                    print(f"❌ Lock file update failed: {result2.stderr}")
                    return False

        except subprocess.TimeoutExpired:
            print("⚠️ Lock file update timed out")
            return False
        except Exception as e:
            print(f"❌ Lock file update failed: {e}")
            return False

    def test_installation(self) -> bool:
        """Test if dependencies can be installed successfully"""
        print("🧪 Testing dependency installation...")

        try:
            # Try to install in a temporary environment
            result = subprocess.run(
                ['uv', 'sync', '--no-dev'],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                print("✅ Dependency installation test passed")
                return True
            else:
                print(f"❌ Installation test failed:\n{result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("⚠️ Installation test timed out")
            return False
        except Exception as e:
            print(f"❌ Installation test failed: {e}")
            return False

    def generate_fix_report(self, backup_path: str) -> str:
        """Generate a report of all fixes applied"""
        report = f"""
# Dependency Fix Report

**Generated**: {datetime.now().isoformat()}
**Backup Location**: {backup_path}

## Fixes Applied

"""
        for i, fix in enumerate(self.fixes_applied, 1):
            report += f"{i}. {fix}\n"

        if not self.fixes_applied:
            report += "No fixes were required - dependencies are already in good state.\n"

        report += f"""

## Next Steps

1. **Verify Changes**: Review the changes made to your dependency files
2. **Test Locally**: Run `uv sync` to test the changes locally
3. **Run Tests**: Execute your test suite to ensure everything works
4. **Commit Changes**: If everything looks good, commit the updated files

## Rollback Instructions

If you need to rollback these changes:

```bash
# Restore from backup
cp {backup_path}/pyproject.toml pyproject.toml
cp {backup_path}/requirements.txt requirements.txt  # if it existed
cp {backup_path}/uv.lock uv.lock  # if it existed

# Re-install dependencies
uv sync
```

## Support

If you encounter issues, check:
- The backup files in `{backup_path}`
- Run `python scripts/validate_dependencies.py` to re-validate
- Check the CI/CD logs for specific error messages
"""
        return report

    def run_all_fixes(self) -> bool:
        """Run all dependency fixes in sequence"""
        print("🚀 Starting automatic dependency fixing...")

        # Create backup first
        backup_path = self.create_backup()

        # Apply fixes in order
        fixes = [
            ("Python Version Mismatch", self.fix_python_version_mismatch),
            ("Version Conflicts", self.fix_version_conflicts),
            ("Dependency Consolidation", self.consolidate_dependencies),
            ("Lock File Update", self.update_lock_file),
            ("Installation Test", self.test_installation),
        ]

        all_passed = True
        for fix_name, fix_func in fixes:
            print(f"\n🔧 Running: {fix_name}")
            try:
                if not fix_func():
                    print(f"❌ {fix_name} failed")
                    all_passed = False
                else:
                    print(f"✅ {fix_name} completed")
            except Exception as e:
                print(f"❌ {fix_name} failed with error: {e}")
                all_passed = False

        # Generate report
        report = self.generate_fix_report(backup_path)
        report_path = self.project_root / "dependency-fix-report.md"
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n📝 Fix report saved to: {report_path}")

        if all_passed:
            print("\n✅ All dependency fixes completed successfully!")
        else:
            print("\n⚠️ Some fixes failed - check the logs above")

        print(f"\n💾 Backup available at: {backup_path}")
        print(f"🔍 Applied {len(self.fixes_applied)} fixes")

        return all_passed


def main():
    """Main fix function"""
    fixer = DependencyFixer()

    try:
        success = fixer.run_all_fixes()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Fix process interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fix process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()