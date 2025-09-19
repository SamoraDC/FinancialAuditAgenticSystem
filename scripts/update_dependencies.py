#!/usr/bin/env python3
"""
Automated Dependency Update Script
Safely updates dependencies with comprehensive testing
"""

import json
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
import toml
import tempfile


class DependencyUpdater:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.backup_dir = project_root / "backups" / "dependencies"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> Path:
        """Create backup of current dependency files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)

        # Backup key files
        files_to_backup = ['pyproject.toml', 'uv.lock']

        for file_name in files_to_backup:
            source = self.project_root / file_name
            if source.exists():
                shutil.copy2(source, backup_path / file_name)
                print(f"📋 Backed up {file_name}")

        print(f"💾 Backup created at {backup_path}")
        return backup_path

    def restore_backup(self, backup_path: Path) -> bool:
        """Restore from backup"""
        try:
            for file_path in backup_path.iterdir():
                if file_path.is_file():
                    target = self.project_root / file_path.name
                    shutil.copy2(file_path, target)
                    print(f"🔄 Restored {file_path.name}")
            return True
        except Exception as e:
            print(f"❌ Failed to restore backup: {e}")
            return False

    def get_outdated_packages(self) -> List[Dict]:
        """Get list of outdated packages"""
        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout) if result.stdout.strip() else []
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to get outdated packages: {e}")
            return []

    def update_pyproject_dependency(self, package_name: str, new_version: str) -> bool:
        """Update a dependency version in pyproject.toml"""
        try:
            with open(self.pyproject_path, 'r') as f:
                pyproject = toml.load(f)

            updated = False

            # Update in main dependencies
            if 'project' in pyproject and 'dependencies' in pyproject['project']:
                for i, dep in enumerate(pyproject['project']['dependencies']):
                    if dep.startswith(f"{package_name}>=") or dep.startswith(f"{package_name}=="):
                        pyproject['project']['dependencies'][i] = f"{package_name}>={new_version}"
                        updated = True
                        print(f"📝 Updated {package_name} to {new_version} in main dependencies")

            # Update in optional dependencies
            if 'project' in pyproject and 'optional-dependencies' in pyproject['project']:
                for group_name, deps in pyproject['project']['optional-dependencies'].items():
                    for i, dep in enumerate(deps):
                        if dep.startswith(f"{package_name}>=") or dep.startswith(f"{package_name}=="):
                            pyproject['project']['optional-dependencies'][group_name][i] = f"{package_name}>={new_version}"
                            updated = True
                            print(f"📝 Updated {package_name} to {new_version} in {group_name} dependencies")

            if updated:
                with open(self.pyproject_path, 'w') as f:
                    toml.dump(pyproject, f)

            return updated

        except Exception as e:
            print(f"❌ Failed to update pyproject.toml: {e}")
            return False

    def run_tests(self) -> bool:
        """Run test suite to verify updates"""
        print("🧪 Running tests to verify updates...")

        try:
            # Install updated dependencies
            result = subprocess.run(['uv', 'sync'], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to install updated dependencies: {result.stderr}")
                return False

            # Run dependency validation
            result = subprocess.run(['python', 'scripts/validate_dependencies.py'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Dependency validation failed: {result.stderr}")
                return False

            # Run backend tests
            result = subprocess.run([
                'uv', 'run', 'pytest', 'backend/tests/',
                '--tb=short', '-x'  # Stop on first failure
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                print("✅ All tests passed!")
                return True
            else:
                print(f"❌ Tests failed: {result.stdout}\n{result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Test execution failed: {e}")
            return False

    def safe_update_single_package(self, package: Dict) -> bool:
        """Safely update a single package with rollback capability"""
        package_name = package['name']
        current_version = package['version']
        new_version = package['latest_version']

        print(f"\n🔄 Updating {package_name}: {current_version} → {new_version}")

        # Create backup before update
        backup_path = self.create_backup()

        try:
            # Update pyproject.toml
            if not self.update_pyproject_dependency(package_name, new_version):
                print(f"⚠️  No dependency entry found for {package_name} in pyproject.toml")
                return False

            # Update lock file
            result = subprocess.run(['uv', 'lock'], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to update lock file: {result.stderr}")
                self.restore_backup(backup_path)
                return False

            # Run tests
            if self.run_tests():
                print(f"✅ Successfully updated {package_name}")
                return True
            else:
                print(f"❌ Tests failed after updating {package_name}, rolling back...")
                self.restore_backup(backup_path)
                return False

        except Exception as e:
            print(f"❌ Error updating {package_name}: {e}")
            self.restore_backup(backup_path)
            return False

    def batch_update_packages(self, packages: List[Dict], max_concurrent: int = 3) -> Dict:
        """Update packages in small batches to minimize risk"""
        print(f"\n📦 Updating {len(packages)} packages in batches of {max_concurrent}")

        results = {
            "successful": [],
            "failed": [],
            "skipped": []
        }

        # Sort packages by risk (prioritize patches over minor/major updates)
        def update_risk(pkg):
            current = pkg['version'].split('.')
            latest = pkg['latest_version'].split('.')

            if len(current) >= 3 and len(latest) >= 3:
                if current[0] != latest[0]:  # Major version change
                    return 3
                elif current[1] != latest[1]:  # Minor version change
                    return 2
                else:  # Patch version change
                    return 1
            return 2  # Default to medium risk

        sorted_packages = sorted(packages, key=update_risk)

        for i in range(0, len(sorted_packages), max_concurrent):
            batch = sorted_packages[i:i + max_concurrent]
            print(f"\n🔄 Processing batch {i//max_concurrent + 1}")

            batch_backup = self.create_backup()

            try:
                # Update all packages in batch
                batch_success = True
                for package in batch:
                    if not self.update_pyproject_dependency(package['name'], package['latest_version']):
                        print(f"⚠️  Skipped {package['name']} - not found in pyproject.toml")
                        results["skipped"].append(package['name'])
                        continue

                # Update lock file for entire batch
                result = subprocess.run(['uv', 'lock'], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ Batch lock update failed: {result.stderr}")
                    self.restore_backup(batch_backup)
                    for pkg in batch:
                        results["failed"].append(pkg['name'])
                    continue

                # Test batch
                if self.run_tests():
                    print(f"✅ Batch update successful")
                    for pkg in batch:
                        results["successful"].append(pkg['name'])
                else:
                    print(f"❌ Batch tests failed, rolling back...")
                    self.restore_backup(batch_backup)
                    for pkg in batch:
                        results["failed"].append(pkg['name'])

            except Exception as e:
                print(f"❌ Batch update error: {e}")
                self.restore_backup(batch_backup)
                for pkg in batch:
                    results["failed"].append(pkg['name'])

        return results

    def conservative_update(self) -> Dict:
        """Conservative update approach - only patch versions"""
        print("🛡️  Running conservative updates (patch versions only)")

        outdated = self.get_outdated_packages()
        if not outdated:
            print("✅ No outdated packages found")
            return {"successful": [], "failed": [], "skipped": []}

        # Filter to patch-only updates
        patch_updates = []
        for pkg in outdated:
            current_parts = pkg['version'].split('.')
            latest_parts = pkg['latest_version'].split('.')

            if (len(current_parts) >= 3 and len(latest_parts) >= 3 and
                current_parts[0] == latest_parts[0] and  # Same major
                current_parts[1] == latest_parts[1]):    # Same minor
                patch_updates.append(pkg)

        if not patch_updates:
            print("ℹ️  No patch-level updates available")
            return {"successful": [], "failed": [], "skipped": []}

        print(f"📦 Found {len(patch_updates)} patch updates")
        return self.batch_update_packages(patch_updates)

    def aggressive_update(self) -> Dict:
        """Aggressive update approach - all updates"""
        print("⚡ Running aggressive updates (all versions)")

        outdated = self.get_outdated_packages()
        if not outdated:
            print("✅ No outdated packages found")
            return {"successful": [], "failed": [], "skipped": []}

        print(f"📦 Found {len(outdated)} updates available")
        return self.batch_update_packages(outdated, max_concurrent=1)  # More careful with aggressive updates

    def generate_update_report(self, results: Dict) -> None:
        """Generate update report"""
        timestamp = datetime.now()
        report = {
            "timestamp": timestamp.isoformat(),
            "results": results,
            "summary": {
                "total_attempted": len(results["successful"]) + len(results["failed"]),
                "successful": len(results["successful"]),
                "failed": len(results["failed"]),
                "skipped": len(results["skipped"])
            }
        }

        report_file = self.backup_dir / f"update_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📋 Update Report:")
        print(f"  ✅ Successful: {report['summary']['successful']}")
        print(f"  ❌ Failed: {report['summary']['failed']}")
        print(f"  ⏭️  Skipped: {report['summary']['skipped']}")
        print(f"  📄 Report saved to: {report_file}")


def main():
    """Main update function"""
    project_root = Path(__file__).parent.parent
    updater = DependencyUpdater(project_root)

    print("🚀 Dependency Update Automation")
    print("=" * 50)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "conservative":
            results = updater.conservative_update()
        elif mode == "aggressive":
            results = updater.aggressive_update()
        elif mode == "single" and len(sys.argv) > 2:
            package_name = sys.argv[2]
            # Find package in outdated list
            outdated = updater.get_outdated_packages()
            package = next((p for p in outdated if p['name'] == package_name), None)
            if package:
                success = updater.safe_update_single_package(package)
                results = {"successful": [package_name] if success else [],
                          "failed": [] if success else [package_name],
                          "skipped": []}
            else:
                print(f"❌ Package {package_name} not found in outdated list")
                return
        else:
            print("Usage: python scripts/update_dependencies.py [conservative|aggressive|single <package_name>]")
            return
    else:
        # Default to conservative
        results = updater.conservative_update()

    updater.generate_update_report(results)
    print("=" * 50)
    print("✅ Dependency update completed!")


if __name__ == "__main__":
    main()