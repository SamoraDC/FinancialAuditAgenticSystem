#!/usr/bin/env python3
"""
Dependency Monitor Script
Continuously monitors and maintains dependency health
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import toml


class DependencyMonitor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.reports_dir = project_root / "reports" / "dependencies"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def check_outdated_packages(self) -> Dict[str, Any]:
        """Check for outdated packages"""
        print("📦 Checking for outdated packages...")

        try:
            # Use pip list --outdated to check for updates
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True, check=True
            )

            outdated = json.loads(result.stdout) if result.stdout.strip() else []

            report = {
                "timestamp": datetime.now().isoformat(),
                "outdated_count": len(outdated),
                "packages": outdated
            }

            # Save report
            report_file = self.reports_dir / f"outdated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            if outdated:
                print(f"⚠️  Found {len(outdated)} outdated packages:")
                for pkg in outdated:
                    print(f"  - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
            else:
                print("✅ All packages are up to date")

            return report

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to check outdated packages: {e}")
            return {"error": str(e)}

    def security_audit(self) -> Dict[str, Any]:
        """Run security audit on dependencies"""
        print("🔒 Running security audit...")

        try:
            # Try pip-audit first
            result = subprocess.run(
                ['pip-audit', '--format=json', '--progress-spinner=off'],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                audit_result = json.loads(result.stdout) if result.stdout.strip() else {}
            else:
                # Fallback to safety if pip-audit fails
                try:
                    result = subprocess.run(
                        ['safety', 'check', '--json'],
                        capture_output=True, text=True
                    )
                    audit_result = json.loads(result.stdout) if result.stdout.strip() else {}
                except (subprocess.CalledProcessError, FileNotFoundError):
                    audit_result = {"error": "No security audit tools available"}

            report = {
                "timestamp": datetime.now().isoformat(),
                "tool": "pip-audit" if result.returncode == 0 else "safety",
                "vulnerabilities": audit_result.get("vulnerabilities", []),
                "vulnerability_count": len(audit_result.get("vulnerabilities", []))
            }

            # Save report
            report_file = self.reports_dir / f"security_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            if report["vulnerability_count"] > 0:
                print(f"🚨 Found {report['vulnerability_count']} security vulnerabilities!")
                for vuln in report["vulnerabilities"][:5]:  # Show first 5
                    print(f"  - {vuln.get('name', 'Unknown')}: {vuln.get('title', 'No description')}")
            else:
                print("✅ No security vulnerabilities found")

            return report

        except Exception as e:
            print(f"❌ Security audit failed: {e}")
            return {"error": str(e)}

    def analyze_dependency_tree(self) -> Dict[str, Any]:
        """Analyze dependency tree for conflicts and issues"""
        print("🌳 Analyzing dependency tree...")

        try:
            # Get dependency tree
            result = subprocess.run(
                ['pipdeptree', '--json'],
                capture_output=True, text=True, check=True
            )

            tree_data = json.loads(result.stdout)

            # Analyze for conflicts
            conflicts = []
            dependency_counts = {}

            for package in tree_data:
                pkg_name = package['package']['package_name']
                dependencies = package.get('dependencies', [])

                dependency_counts[pkg_name] = len(dependencies)

                # Check for potential conflicts (packages with many dependencies)
                if len(dependencies) > 10:
                    conflicts.append({
                        "package": pkg_name,
                        "dependency_count": len(dependencies),
                        "type": "high_dependency_count"
                    })

            report = {
                "timestamp": datetime.now().isoformat(),
                "total_packages": len(tree_data),
                "conflicts": conflicts,
                "dependency_stats": {
                    "max_dependencies": max(dependency_counts.values()) if dependency_counts else 0,
                    "avg_dependencies": sum(dependency_counts.values()) / len(dependency_counts) if dependency_counts else 0,
                    "packages_with_many_deps": len([c for c in dependency_counts.values() if c > 5])
                }
            }

            # Save report
            report_file = self.reports_dir / f"tree_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"📊 Analyzed {report['total_packages']} packages")
            if conflicts:
                print(f"⚠️  Found {len(conflicts)} potential issues:")
                for conflict in conflicts[:3]:
                    print(f"  - {conflict['package']}: {conflict['dependency_count']} dependencies")

            return report

        except subprocess.CalledProcessError as e:
            print(f"❌ Dependency tree analysis failed: {e}")
            return {"error": str(e)}
        except FileNotFoundError:
            print("⚠️  pipdeptree not available, skipping tree analysis")
            return {"error": "pipdeptree not available"}

    def check_license_compliance(self) -> Dict[str, Any]:
        """Check license compliance of dependencies"""
        print("📄 Checking license compliance...")

        try:
            result = subprocess.run(
                ['pip-licenses', '--format=json'],
                capture_output=True, text=True, check=True
            )

            licenses = json.loads(result.stdout)

            # Define problematic licenses
            problematic_licenses = ['GPL-3.0', 'AGPL-3.0', 'LGPL-3.0']
            unknown_licenses = ['UNKNOWN', 'UNLICENSED', '']

            issues = []
            for pkg in licenses:
                license_name = pkg.get('License', '').upper()
                if any(prob in license_name for prob in problematic_licenses):
                    issues.append({
                        "package": pkg['Name'],
                        "license": pkg['License'],
                        "type": "restrictive_license"
                    })
                elif license_name in unknown_licenses or not license_name:
                    issues.append({
                        "package": pkg['Name'],
                        "license": pkg['License'],
                        "type": "unknown_license"
                    })

            report = {
                "timestamp": datetime.now().isoformat(),
                "total_packages": len(licenses),
                "license_issues": issues,
                "issue_count": len(issues)
            }

            # Save report
            report_file = self.reports_dir / f"licenses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            if issues:
                print(f"⚠️  Found {len(issues)} license issues:")
                for issue in issues[:5]:
                    print(f"  - {issue['package']}: {issue['license']} ({issue['type']})")
            else:
                print("✅ No license compliance issues found")

            return report

        except subprocess.CalledProcessError as e:
            print(f"❌ License check failed: {e}")
            return {"error": str(e)}
        except FileNotFoundError:
            print("⚠️  pip-licenses not available, skipping license check")
            return {"error": "pip-licenses not available"}

    def generate_summary_report(self, reports: Dict[str, Any]) -> None:
        """Generate comprehensive summary report"""
        print("\n📋 Generating summary report...")

        timestamp = datetime.now()
        summary = {
            "timestamp": timestamp.isoformat(),
            "project": "Financial Audit Agentic System",
            "monitoring_results": reports,
            "recommendations": []
        }

        # Generate recommendations based on findings
        if reports.get("outdated", {}).get("outdated_count", 0) > 0:
            summary["recommendations"].append({
                "type": "update",
                "message": f"Update {reports['outdated']['outdated_count']} outdated packages",
                "priority": "medium"
            })

        if reports.get("security", {}).get("vulnerability_count", 0) > 0:
            summary["recommendations"].append({
                "type": "security",
                "message": f"Address {reports['security']['vulnerability_count']} security vulnerabilities",
                "priority": "high"
            })

        if reports.get("licenses", {}).get("issue_count", 0) > 0:
            summary["recommendations"].append({
                "type": "license",
                "message": f"Review {reports['licenses']['issue_count']} license compliance issues",
                "priority": "medium"
            })

        # Save summary
        summary_file = self.reports_dir / f"summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create human-readable report
        readme_content = f"""# Dependency Monitoring Report

**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

"""

        if not summary["recommendations"]:
            readme_content += "✅ **All dependency checks passed!**\n\n"
        else:
            readme_content += f"⚠️  **Found {len(summary['recommendations'])} recommendations:**\n\n"
            for rec in summary["recommendations"]:
                priority_emoji = "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🟢"
                readme_content += f"- {priority_emoji} {rec['message']}\n"

        readme_content += f"""
## Detailed Results

### Outdated Packages
- **Count:** {reports.get("outdated", {}).get("outdated_count", "N/A")}
- **Status:** {"❌ Updates needed" if reports.get("outdated", {}).get("outdated_count", 0) > 0 else "✅ Up to date"}

### Security Vulnerabilities
- **Count:** {reports.get("security", {}).get("vulnerability_count", "N/A")}
- **Status:** {"🚨 Vulnerabilities found" if reports.get("security", {}).get("vulnerability_count", 0) > 0 else "✅ No vulnerabilities"}

### License Compliance
- **Issues:** {reports.get("licenses", {}).get("issue_count", "N/A")}
- **Status:** {"⚠️  Issues found" if reports.get("licenses", {}).get("issue_count", 0) > 0 else "✅ Compliant"}

### Dependency Tree
- **Total Packages:** {reports.get("tree", {}).get("total_packages", "N/A")}
- **Conflicts:** {len(reports.get("tree", {}).get("conflicts", []))}

## Next Steps

1. Review detailed JSON reports in the `reports/dependencies/` directory
2. Address high-priority recommendations first
3. Run `python scripts/validate_dependencies.py` to verify fixes
4. Update this report by running `python scripts/dependency_monitor.py`

---
*Automated dependency monitoring for Financial Audit Agentic System*
"""

        readme_file = self.reports_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)

        print(f"📋 Summary report saved to {summary_file}")
        print(f"📄 Human-readable report saved to {readme_file}")

    def run_all_checks(self) -> None:
        """Run all dependency monitoring checks"""
        print("🚀 Starting comprehensive dependency monitoring")
        print("=" * 60)

        reports = {}

        # Run all checks
        reports["outdated"] = self.check_outdated_packages()
        print()

        reports["security"] = self.security_audit()
        print()

        reports["tree"] = self.analyze_dependency_tree()
        print()

        reports["licenses"] = self.check_license_compliance()
        print()

        # Generate summary
        self.generate_summary_report(reports)

        print("=" * 60)
        print("✅ Dependency monitoring completed!")


def main():
    """Main monitoring function"""
    project_root = Path(__file__).parent.parent
    monitor = DependencyMonitor(project_root)

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "outdated":
            monitor.check_outdated_packages()
        elif command == "security":
            monitor.security_audit()
        elif command == "tree":
            monitor.analyze_dependency_tree()
        elif command == "licenses":
            monitor.check_license_compliance()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: outdated, security, tree, licenses")
            sys.exit(1)
    else:
        monitor.run_all_checks()


if __name__ == "__main__":
    main()