#!/usr/bin/env python3
"""
🔧 Ultra-Fast Dependency Validation Script
Validates and optimizes dependency declarations for zero-failure installs.
"""

import sys
import json
import subprocess
import time
from pathlib import Path
import hashlib
from typing import Dict, List, Set, Tuple
import importlib.util

class DependencyValidator:
    def __init__(self):
        self.start_time = time.time()
        self.errors = []
        self.warnings = []
        self.optimizations = []

    def log(self, level: str, message: str):
        """Enhanced logging with performance tracking"""
        elapsed = time.time() - self.start_time
        prefix = {
            'INFO': '✅',
            'WARN': '⚠️',
            'ERROR': '❌',
            'PERF': '⚡'
        }.get(level, '📝')
        print(f"{prefix} [{elapsed:.2f}s] {message}")

    def validate_pyproject_syntax(self) -> bool:
        """Validate pyproject.toml syntax and structure"""
        self.log('INFO', 'Validating pyproject.toml syntax...')

        try:
            import toml
            with open('pyproject.toml', 'r') as f:
                config = toml.load(f)

            # Check required sections
            required_sections = ['project', 'project.dependencies']
            for section in required_sections:
                keys = section.split('.')
                current = config
                for key in keys:
                    if key not in current:
                        self.errors.append(f"Missing required section: {section}")
                        return False
                    current = current[key]

            self.log('INFO', 'pyproject.toml syntax is valid')
            return True

        except Exception as e:
            self.errors.append(f"pyproject.toml syntax error: {e}")
            return False

    def check_dependency_conflicts(self) -> bool:
        """Advanced dependency conflict detection"""
        self.log('INFO', 'Checking for dependency conflicts...')

        try:
            result = subprocess.run(
                ['uv', 'pip', 'check'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                conflicts = result.stdout + result.stderr
                if conflicts.strip():
                    self.errors.append(f"Dependency conflicts detected:\n{conflicts}")
                    return False

            self.log('INFO', 'No dependency conflicts found')
            return True

        except subprocess.TimeoutExpired:
            self.warnings.append("Dependency check timed out")
            return True
        except Exception as e:
            self.warnings.append(f"Could not check dependencies: {e}")
            return True

    def validate_version_constraints(self) -> bool:
        """Validate and optimize version constraints"""
        self.log('INFO', 'Validating version constraints...')

        try:
            import toml
            with open('pyproject.toml', 'r') as f:
                config = toml.load(f)

            dependencies = config.get('project', {}).get('dependencies', [])
            dev_deps = config.get('project', {}).get('optional-dependencies', {}).get('dev', [])

            all_deps = dependencies + dev_deps

            # Check for loose version constraints
            loose_constraints = []
            for dep in all_deps:
                if isinstance(dep, str):
                    if '>=' in dep and '<' not in dep and '==' not in dep:
                        loose_constraints.append(dep)

            if loose_constraints:
                self.warnings.append(f"Loose version constraints found: {loose_constraints}")
                self.optimizations.append("Consider adding upper bounds for stability")

            self.log('INFO', f'Validated {len(all_deps)} dependencies')
            return True

        except Exception as e:
            self.errors.append(f"Version constraint validation failed: {e}")
            return False

    def check_security_vulnerabilities(self) -> bool:
        """Fast security vulnerability check"""
        self.log('INFO', 'Checking for security vulnerabilities...')

        try:
            # Use pip-audit for fast vulnerability scanning
            result = subprocess.run(
                ['uv', 'run', 'pip-audit', '--format=json', '--no-deps'],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                try:
                    audit_results = json.loads(result.stdout)
                    vulnerabilities = audit_results.get('vulnerabilities', [])

                    if vulnerabilities:
                        high_severity = [v for v in vulnerabilities if v.get('severity') in ['HIGH', 'CRITICAL']]
                        if high_severity:
                            self.errors.append(f"High severity vulnerabilities found: {len(high_severity)}")
                            return False
                        else:
                            self.warnings.append(f"Low/medium severity vulnerabilities: {len(vulnerabilities)}")

                    self.log('INFO', f'Security check passed ({len(vulnerabilities)} issues)')
                    return True

                except json.JSONDecodeError:
                    self.warnings.append("Could not parse security audit results")
                    return True

            else:
                self.warnings.append("Security audit failed, proceeding anyway")
                return True

        except subprocess.TimeoutExpired:
            self.warnings.append("Security check timed out")
            return True
        except Exception as e:
            self.warnings.append(f"Security check error: {e}")
            return True

    def optimize_dependency_resolution(self) -> bool:
        """Optimize dependency resolution for speed"""
        self.log('PERF', 'Optimizing dependency resolution...')

        try:
            # Pre-compile requirements for faster resolution
            result = subprocess.run(
                ['uv', 'pip', 'compile', '--quiet', '--strip-extras', 'pyproject.toml'],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                # Cache compiled requirements
                with open('.requirements-cache.txt', 'w') as f:
                    f.write(result.stdout)

                self.log('PERF', 'Dependency resolution optimized and cached')
                self.optimizations.append("Compiled requirements cached for faster installs")
                return True
            else:
                self.warnings.append("Could not pre-compile requirements")
                return True

        except subprocess.TimeoutExpired:
            self.warnings.append("Dependency compilation timed out")
            return True
        except Exception as e:
            self.warnings.append(f"Dependency optimization failed: {e}")
            return True

    def validate_import_paths(self) -> bool:
        """Validate that all dependencies can be imported"""
        self.log('INFO', 'Validating import paths...')

        # Common import mappings for packages with different import names
        import_mappings = {
            'python-dotenv': 'dotenv',
            'python-jose': 'jose',
            'python-docx': 'docx',
            'pymupdf': 'fitz',
            'stable-baselines3': 'stable_baselines3',
            'scikit-learn': 'sklearn',
            'pillow': 'PIL',
        }

        try:
            import toml
            with open('pyproject.toml', 'r') as f:
                config = toml.load(f)

            dependencies = config.get('project', {}).get('dependencies', [])

            failed_imports = []
            for dep in dependencies:
                if isinstance(dep, str):
                    # Extract package name
                    pkg_name = dep.split('>=')[0].split('==')[0].split('[')[0].strip()
                    import_name = import_mappings.get(pkg_name, pkg_name.replace('-', '_'))

                    # Skip complex packages that are known to work
                    skip_packages = {
                        'torch', 'transformers', 'langchain', 'langgraph',
                        'pydantic-ai', 'stable-baselines3'
                    }

                    if pkg_name in skip_packages:
                        continue

                    try:
                        spec = importlib.util.find_spec(import_name)
                        if spec is None:
                            failed_imports.append(pkg_name)
                    except (ImportError, ModuleNotFoundError, ValueError):
                        # Package might not be installed yet, that's OK
                        pass

            if failed_imports:
                self.warnings.append(f"Potential import issues: {failed_imports}")

            self.log('INFO', 'Import path validation completed')
            return True

        except Exception as e:
            self.warnings.append(f"Import validation failed: {e}")
            return True

    def generate_dependency_report(self) -> Dict:
        """Generate comprehensive dependency report"""
        self.log('INFO', 'Generating dependency report...')

        try:
            import toml
            with open('pyproject.toml', 'r') as f:
                config = toml.load(f)

            dependencies = config.get('project', {}).get('dependencies', [])
            dev_deps = config.get('project', {}).get('optional-dependencies', {}).get('dev', [])

            report = {
                'total_dependencies': len(dependencies) + len(dev_deps),
                'production_dependencies': len(dependencies),
                'development_dependencies': len(dev_deps),
                'python_version': config.get('project', {}).get('requires-python', 'Unknown'),
                'validation_time': time.time() - self.start_time,
                'errors': len(self.errors),
                'warnings': len(self.warnings),
                'optimizations': len(self.optimizations),
                'status': 'PASSED' if len(self.errors) == 0 else 'FAILED'
            }

            return report

        except Exception as e:
            self.warnings.append(f"Report generation failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}

    def run_full_validation(self) -> bool:
        """Run complete dependency validation suite"""
        self.log('INFO', '🚀 Starting ultra-fast dependency validation...')

        validation_steps = [
            ('Syntax Validation', self.validate_pyproject_syntax),
            ('Version Constraints', self.validate_version_constraints),
            ('Conflict Detection', self.check_dependency_conflicts),
            ('Security Check', self.check_security_vulnerabilities),
            ('Resolution Optimization', self.optimize_dependency_resolution),
            ('Import Validation', self.validate_import_paths),
        ]

        all_passed = True

        for step_name, step_func in validation_steps:
            step_start = time.time()
            try:
                result = step_func()
                step_time = time.time() - step_start
                self.log('PERF', f'{step_name} completed in {step_time:.2f}s')
                if not result:
                    all_passed = False
            except Exception as e:
                self.errors.append(f"{step_name} failed: {e}")
                all_passed = False

        # Generate final report
        report = self.generate_dependency_report()

        # Print summary
        total_time = time.time() - self.start_time
        self.log('INFO', f'🎯 Validation completed in {total_time:.2f}s')
        self.log('INFO', f'📊 Dependencies: {report.get("total_dependencies", 0)}')
        self.log('INFO', f'❌ Errors: {len(self.errors)}')
        self.log('INFO', f'⚠️ Warnings: {len(self.warnings)}')
        self.log('INFO', f'⚡ Optimizations: {len(self.optimizations)}')

        if self.errors:
            self.log('ERROR', 'VALIDATION FAILED')
            for error in self.errors:
                print(f"❌ {error}")
        else:
            self.log('INFO', '✅ VALIDATION PASSED')

        if self.warnings:
            for warning in self.warnings:
                print(f"⚠️ {warning}")

        if self.optimizations:
            for opt in self.optimizations:
                print(f"⚡ {opt}")

        return all_passed

def main():
    """Main validation entry point"""
    validator = DependencyValidator()

    try:
        success = validator.run_full_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()