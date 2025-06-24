#!/usr/bin/env python3
"""
Test script for Docker container builds with different arguments.
This script tests the Dockerfile in the tests directory with various configurations.
"""

import subprocess
import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DockerBuildTester:
    def __init__(self, dockerfile_path: str, base_dir: str, log_dir: Optional[str] = None):
        self.dockerfile_path = Path(dockerfile_path)
        self.base_dir = Path(base_dir)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.test_results = []

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)

    def build_image(self, build_args: Dict[str, str], image_name: str) -> Tuple[bool, str]:
        """Build Docker image with given arguments."""
        try:
            # Execute hardcoded pre-build command using TEST_APP_NAME
            test_app_name = build_args.get("TEST_APP_NAME", "frame_sampling")
            pre_build_command = (
                f"python ../main.py {test_app_name}/app.yaml "
                f"-o {test_app_name} "
                f"-c {test_app_name}/processors.py "
                f"--server-type serverless -t"
            )

            logger.info(f"🔧 Executing pre-build command: {pre_build_command}")
            pre_build_result = subprocess.run(
                pre_build_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for pre-build
            )

            if pre_build_result.returncode != 0:
                error_msg = f"Pre-build command failed: {pre_build_result.stderr}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            logger.info("✅ Pre-build command completed successfully")
            if pre_build_result.stdout:
                logger.info(f"Pre-build output: {pre_build_result.stdout}")

            cmd = [
                "docker", "build",
                "-f", str(self.dockerfile_path),
                "-t", image_name
            ]

            # Add build arguments
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

            # Add context directory
            cmd.append(str(self.base_dir))

            logger.info(f"Building image: {image_name}")
            logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ Successfully built image: {image_name}")
                return True, result.stdout
            else:
                logger.error(f"❌ Failed to build image: {image_name}")
                logger.error(f"Error: {result.stderr}")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Build timed out for image: {image_name}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Exception during build: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def run_prerequisite_script(self, script_command: str, test_id: int) -> Tuple[bool, str]:
        """Run a prerequisite script before testing the docker container."""
        if not script_command:
            return True, "No prerequisite script specified"

        log_file = self.log_dir / f"prerequisite_{test_id}.log"

        try:
            logger.info(f"🔧 Running prerequisite script: {script_command}")
            logger.info(f"📄 Prerequisite logs will be saved to: {log_file}")

            # Run the prerequisite script
            result = subprocess.run(
                script_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for prerequisite scripts
            )

            # Save prerequisite script logs
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
                f.write("\n=== END LOG ===\n")

            if result.returncode == 0:
                logger.info(f"✅ Prerequisite script completed successfully")
                if result.stdout:
                    logger.info(f"Prerequisite output: {result.stdout}")
                return True, result.stdout
            else:
                error_msg = f"Prerequisite script failed with return code {result.returncode}"
                logger.error(f"❌ {error_msg}")
                logger.error(f"Prerequisite stderr: {result.stderr}")
                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = f"Prerequisite script timed out after 300 seconds"
            logger.error(f"❌ {error_msg}")

            # Save timeout log
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Status: TIMEOUT\n")
                f.write(f"Timeout: 300 seconds\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n=== END LOG ===\n")

            return False, error_msg
        except Exception as e:
            error_msg = f"Exception during prerequisite script execution: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # Save exception log
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Status: EXCEPTION\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n=== END LOG ===\n")

            return False, error_msg

    def test_image(self, image_name: str, test_config: Dict, test_id: int) -> Tuple[bool, str, str]:
        """Test the built image by running it and capture logs."""
        log_file = self.log_dir / f"test_{test_id}_{image_name.replace(':', '_')}.log"

        # Get timeout from test config, default to 10 seconds
        timeout = test_config.get("timeout", 10)

        try:
            # Check if image exists
            result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return False, f"Image {image_name} not found", ""

            # Run prerequisite script if specified
            prerequisite_script = test_config.get("prerequisite_script")
            if prerequisite_script:
                prerequisite_success, prerequisite_output = self.run_prerequisite_script(
                    prerequisite_script, test_id
                )
                if not prerequisite_success:
                    return False, f"Prerequisite script failed: {prerequisite_output}", ""

            # Run the container with test configuration
            cmd = ["docker", "run", "--rm", "--network=host"]

            # Add environment variables if specified
            if "env" in test_config:
                for key, value in test_config["env"].items():
                    cmd.extend(["-e", f"{key}={value}"])

            # Add volume mounts if specified
            if "volumes" in test_config:
                logger.info(f"📁 Processing volumes: {test_config['volumes']}")
                for host_path, container_path in test_config["volumes"].items():
                    # Convert relative path to absolute path based on current working directory
                    if not os.path.isabs(host_path):
                        abs_host_path = os.path.abspath(host_path)
                    else:
                        abs_host_path = host_path

                    volume_arg = f"{abs_host_path}:{container_path}"
                    cmd.extend(["-v", volume_arg])
                    logger.info(f"📁 Adding volume mount: {volume_arg}")
            else:
                logger.info("📁 No volumes specified in test config")

            cmd.append(image_name)

            # Add command arguments if specified
            if "cmd" in test_config:
                cmd.extend(test_config["cmd"])

            logger.info(f"Testing image: {image_name}")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Timeout: {timeout} seconds")
            logger.info(f"Logs will be saved to: {log_file}")

            # Log the complete command for debugging
            logger.info("🔍 Complete docker run command:")
            logger.info(f"   {' '.join(cmd)}")

            # Check if error detection is enabled
            error_detection_config = test_config.get("error_detection", {})
            error_detection_enabled = error_detection_config.get("enabled", False)

            if error_detection_enabled:
                # Capture output for error detection
                logger.info("🔍 Error detection enabled - capturing output for analysis")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout  # Use configurable timeout
                )
                stdout_output = result.stdout
                stderr_output = result.stderr
            else:
                # Stream output to host stdout in real-time
                logger.info("📺 Error detection disabled - streaming output to host stdout")
                logger.info("=" * 80)
                logger.info(f"CONTAINER OUTPUT FOR: {image_name}")
                logger.info("=" * 80)

                # Run container and stream output to host stdout in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True
                )

                # Collect output while streaming to stdout
                stdout_output = ""
                try:
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            print(output, end='', flush=True)
                            stdout_output += output

                    # Wait for process to complete and get return code
                    return_code = process.poll()
                    stderr_output = ""

                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    return_code = process.poll()
                    stderr_output = ""
                    logger.info("\n⚠️  Process interrupted by user")

                # Create result object for consistency
                class Result:
                    def __init__(self, returncode, stdout, stderr):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                result = Result(return_code, stdout_output, stderr_output)

                logger.info("=" * 80)
                logger.info(f"END CONTAINER OUTPUT FOR: {image_name}")
                logger.info("=" * 80)

            # Save logs to file
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Timeout: {timeout} seconds\n")
                f.write(f"Error Detection: {'Enabled' if error_detection_enabled else 'Disabled'}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
                f.write("\n=== END LOG ===\n")

            # Check for ERROR patterns in the output only if error detection is enabled
            if error_detection_enabled:
                error_patterns = error_detection_config.get("patterns", [
                    "ERROR",
                    "Error",
                    "error",
                    "CRITICAL",
                    "Critical",
                    "critical",
                    "FATAL",
                    "Fatal",
                    "fatal",
                    "Exception:",
                    "exception:",
                    "Traceback",
                    "traceback"
                ])

                has_errors = False
                error_lines = []

                # Check stdout for errors
                for line in result.stdout.split('\n'):
                    for pattern in error_patterns:
                        if pattern in line:
                            has_errors = True
                            error_lines.append(f"STDOUT: {line.strip()}")
                            break

                # Check stderr for errors
                for line in result.stderr.split('\n'):
                    for pattern in error_patterns:
                        if pattern in line:
                            has_errors = True
                            error_lines.append(f"STDERR: {line.strip()}")
                            break

                if result.returncode == 0 and not has_errors:
                    logger.info(f"✅ Successfully tested image: {image_name}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return True, result.stdout, str(log_file)
                elif has_errors:
                    error_msg = (
                        f"Test failed due to ERROR logs detected in image: {image_name}"
                    )
                    logger.error(f"❌ {error_msg}")
                    logger.error("Error lines found:")
                    for error_line in error_lines:
                        logger.error(f"  {error_line}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, error_msg, str(log_file)
                else:
                    logger.warning(
                        f"⚠️  Test completed with non-zero return code for image: {image_name}"
                    )
                    logger.warning(f"Return code: {result.returncode}")
                    logger.warning(f"Output: {result.stdout}")
                    logger.warning(f"Error: {result.stderr}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, f"Container exited with code {result.returncode}", str(log_file)
            else:
                # When error detection is disabled, only check return code
                if result.returncode == 0:
                    logger.info(f"✅ Successfully tested image: {image_name}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return True, result.stdout, str(log_file)
                else:
                    logger.warning(
                        f"⚠️  Test completed with non-zero return code for image: {image_name}"
                    )
                    logger.warning(f"Return code: {result.returncode}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, f"Container exited with code {result.returncode}", str(log_file)

        except subprocess.TimeoutExpired:
            error_msg = f"Test timed out after {timeout} seconds for image: {image_name}"
            logger.error(f"❌ {error_msg}")

            # Save timeout log
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Status: TIMEOUT\n")
                f.write(f"Timeout: {timeout} seconds\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n=== END LOG ===\n")

            return False, error_msg, str(log_file)
        except Exception as e:
            error_msg = f"Exception during test: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # Save exception log
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Status: EXCEPTION\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n=== END LOG ===\n")

            return False, error_msg, str(log_file)

    def cleanup_prerequisite_script(self, test_config: Dict, test_id: int) -> bool:
        """Clean up resources created by prerequisite scripts."""
        prerequisite_script = test_config.get("prerequisite_script")
        if not prerequisite_script:
            return True

        try:
            # Check if the script is the RTSP server setup script
            if "setup_rtsp_server.sh" in prerequisite_script:
                logger.info("🧹 Cleaning up RTSP server...")
                cleanup_result = subprocess.run(
                    "./setup_rtsp_server.sh --kill",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if cleanup_result.returncode == 0:
                    logger.info("✅ RTSP server cleanup completed")
                    return True
                else:
                    logger.warning(f"⚠️  RTSP server cleanup failed: {cleanup_result.stderr}")
                    return False

            # Add more cleanup logic for other prerequisite scripts here
            logger.info("🧹 No specific cleanup needed for prerequisite script")
            return True

        except Exception as e:
            logger.warning(f"⚠️  Exception during prerequisite cleanup: {str(e)}")
            return False

    def cleanup_image(self, image_name: str) -> bool:
        """Remove the test image."""
        try:
            result = subprocess.run(
                ["docker", "rmi", image_name],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"🧹 Cleaned up image: {image_name}")
                return True
            else:
                logger.warning(f"⚠️  Failed to cleanup image: {image_name}")
                return False

        except Exception as e:
            logger.warning(f"⚠️  Exception during cleanup: {str(e)}")
            return False

    def run_test_suite(self, test_configs: List[Dict], cleanup: bool = True, gitlab_token: Optional[str] = None) -> Dict:
        """Run a suite of tests with different configurations."""
        results = {
            "total_tests": len(test_configs),
            "passed": 0,
            "failed": 0,
            "results": []
        }

        for i, config in enumerate(test_configs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test {i}/{len(test_configs)}")
            logger.info(f"{'='*60}")

            # Generate unique image name
            image_name = f"test-inference-builder-{i}-{int(time.time())}"

            # Prepare build arguments with gitlab_token if provided
            build_args = config.get("build_args", {}).copy()
            if gitlab_token:
                build_args["GITLAB_TOKEN"] = gitlab_token

            # Build image
            build_success, build_output = self.build_image(build_args, image_name)

            if build_success:
                # Test image
                test_success, test_output, log_file = self.test_image(
                    image_name,
                    config.get("test_config", {}),
                    i
                )

                if test_success:
                    results["passed"] += 1
                    status = "PASSED"
                else:
                    results["failed"] += 1
                    status = "FAILED"
            else:
                results["failed"] += 1
                test_success = False
                test_output = ""
                log_file = ""
                status = "FAILED"

            # Cleanup
            if cleanup:
                self.cleanup_image(image_name)
                # Clean up prerequisite scripts
                self.cleanup_prerequisite_script(config.get("test_config", {}), i)

            # Store result
            test_result = {
                "test_id": i,
                "config": config,
                "status": status,
                "build_success": build_success,
                "test_success": test_success,
                "build_output": build_output,
                "test_output": test_output,
                "log_file": log_file,
                "image_name": image_name
            }

            results["results"].append(test_result)

            logger.info(f"Test {i} result: {status}")

        return results

    def generate_report(self, results: Dict, output_file: Optional[str] = None):
        """Generate a test report."""
        report = {
            "summary": {
                "total_tests": results["total_tests"],
                "passed": results["passed"],
                "failed": results["failed"],
                "success_rate": f"{(results['passed'] / results['total_tests'] * 100):.1f}%" if results["total_tests"] > 0 else "0%"
            },
            "results": results["results"]
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Report saved to: {output_file}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total tests: {results['total_tests']}")
        logger.info(f"Passed: {results['passed']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Success rate: {report['summary']['success_rate']}")

        # Print log file locations
        logger.info(f"\n{'='*60}")
        logger.info("LOG FILES")
        logger.info(f"{'='*60}")
        for result in results["results"]:
            if result.get("log_file"):
                logger.info(f"Test {result['test_id']}: {result['log_file']}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Test Docker builds with different arguments")
    parser.add_argument("--dockerfile", default="Dockerfile", help="Path to Dockerfile")
    parser.add_argument("--base-dir", default=".", help="Base directory for Docker build context")
    parser.add_argument("--config-file", required=True, help="JSON file with test configurations")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--log-dir", default="logs", help="Directory to save container logs")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup images after testing")
    parser.add_argument("--gitlab-token", help="GitLab token for private repos")

    args = parser.parse_args()

    # Initialize tester
    tester = DockerBuildTester(args.dockerfile, args.base_dir, args.log_dir)

    # Load test configurations from file
    with open(args.config_file, 'r') as f:
        test_configs = json.load(f)

    # Run tests
    logger.info(f"Starting test suite with {len(test_configs)} configurations")
    logger.info(f"Logs will be saved to: {args.log_dir}")
    results = tester.run_test_suite(test_configs, cleanup=not args.no_cleanup, gitlab_token=args.gitlab_token)

    # Generate report
    tester.generate_report(results, args.output)

    # Exit with appropriate code
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()