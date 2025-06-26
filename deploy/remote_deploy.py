from typing import List, Dict, Optional
import logging
import subprocess
import yaml
import time
import os
from pathlib import Path
import sys
import json
from dataclasses import dataclass
import argparse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentPreparation:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Helm chart info
        self.chart_url = os.getenv('HELM_CHART_URL', "https://helm.ngc.nvidia.com/eevaigoeixww/dev/charts/tao-cv-app-0.2.2.tgz")
        self.charts_dir = os.path.join(self.base_dir, 'builder/samples/tao/helm/charts')
        self.values_file = os.path.join(self.base_dir, 'builder/samples/tao/helm/tao-cv-app/custom_values.yaml')
        # Update image for helm chart values override
        self.override_image = os.environ.get('OVERRIDE_IMAGE', None)
        self.registry_image = os.environ.get('CI_REGISTRY_IMAGE', None)
        self.flavor = os.environ.get('FLAVOR', None)
        self.commit_sha = os.environ.get('CI_COMMIT_SHA', None)
        # Get SSH private key to connect to deployment host who has pub key in known_hosts
        self.ssh_private_key = os.environ.get('SSH_PRIVATE_KEY', None)
        assert self.ssh_private_key is not None, "SSH_PRIVATE_KEY is not set"
        # Deployment host info
        self.deployment_host_ip = os.environ.get('DEPLOYMENT_HOST_IP', "10.111.53.46")
        self.deployment_host_user = os.environ.get('DEPLOYMENT_HOST_USER', "byin")
        self.deployment_host = f"{self.deployment_host_user}@{self.deployment_host_ip}"
        # kubeconfig info
        self.ci_project_dir = os.environ.get("CI_PROJECT_DIR")
        assert self.ci_project_dir is not None, "CI_PROJECT_DIR is not set"
        self.kubeconfig = os.path.join(self.ci_project_dir, "kubeconfig")

    def prepare(self) -> str:
        """Run preparation steps and return chart path"""
        self.setup_ssh()
        self.get_kubeconfig()
        return self.download_helm_chart()

    def test_network_connectivity(self):
        """Test network connectivity to the target host."""
        import socket
        import subprocess

        host = self.deployment_host_ip  # Your target host (10.111.53.46)
        port = 22  # SSH port

        logging.info(f"Testing network connectivity to {host}:{port}")

        # Test basic socket connection
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            if result == 0:
                logging.info("TCP connection successful")
            else:
                logging.error(f"TCP connection failed with error code: {result}")
        except Exception as e:
            logging.error(f"Socket connection error: {str(e)}")
        finally:
            sock.close()

        # Test with ping
        try:
            ping_cmd = ['ping', '-c', '1', '-W', '5', host]
            result = subprocess.run(ping_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("Ping successful")
            else:
                logging.error(f"Ping failed: {result.stderr}")
        except Exception as e:
            logging.error(f"Ping error: {str(e)}")

        # Test with traceroute if available
        try:
            traceroute_cmd = ['traceroute', '-n', '-w', '2', host]
            result = subprocess.run(traceroute_cmd, capture_output=True, text=True)
            logging.info(f"Traceroute output:\n{result.stdout}")
        except FileNotFoundError:
            logging.warning("Traceroute not available")
        except Exception as e:
            logging.error(f"Traceroute error: {str(e)}")
        
        # Add this to your network testing
        try:
            logging.info("Testing DNS resolution...")
            import socket
            ip_addr = socket.gethostbyname(host)
            logging.info(f"Resolved {host} to {ip_addr}")
        except socket.gaierror as e:
            logging.error(f"DNS resolution failed: {e}")

        # Test direct IP connection
        try:
            logging.info(f"Testing direct IP connection to {ip_addr}")
            result = subprocess.run(['ping', '-c', '1', ip_addr], 
                                capture_output=True, text=True)
            logging.info(f"Ping result: {result.stdout}")
        except Exception as e:
            logging.error(f"Direct IP ping failed: {e}")

    def test_ssh_connection(self):
        """Test SSH key and connection"""
        import subprocess
        ssh_dir = os.path.expanduser('~/.ssh')
        key_path = os.path.join(ssh_dir, 'id_rsa')
        
        # Test 1: Verify key format
        try:
            result = subprocess.run(['ssh-keygen', '-l', '-f', key_path], 
                                capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"SSH key verification successful: {result.stdout.strip()}")
            else:
                logging.error(f"SSH key verification failed: {result.stderr}")
        except Exception as e:
            logging.error(f"Error verifying SSH key: {str(e)}")

        # Test 2: Try SSH connection with verbose output
        try:
            cmd = [
                'ssh',
                '-v',  # verbose output
                '-i', key_path,
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                self.deployment_host,
                'echo "SSH test successful"'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("SSH connection test successful")
            else:
                logging.error("SSH connection test failed")
                logging.error(f"SSH stderr: {result.stderr}")
        except Exception as e:
            logging.error(f"Error testing SSH connection: {str(e)}")

        # Test 3: Check SSH agent
        try:
            result = subprocess.run(['ssh-add', '-l'], capture_output=True, text=True)
            logging.info(f"SSH agent keys: {result.stdout if result.returncode == 0 else 'No keys loaded'}")
        except Exception as e:
            logging.error(f"Error checking SSH agent: {str(e)}")
    
    def verify_kubeconfig(self):
        """Verify kubeconfig file and cluster access"""
        import subprocess
        logger.info("Verifying kubeconfig and cluster access...")
        
        # Check file permissions
        os.chmod(self.kubeconfig, 0o600)  # Ensure correct permissions
        
        # Test connection to cluster
        try:
            cmd = ['kubectl', '--kubeconfig', self.kubeconfig, 'version']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Successfully connected to cluster")
            else:
                logger.error(f"Failed to connect to cluster: {result.stderr}")
        except Exception as e:
            logger.error(f"Error verifying kubeconfig: {str(e)}")
                
        # Test port access
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.deployment_host_ip, 16443))
            if result == 0:
                logger.info("Successfully connected to Kubernetes API port")
            else:
                logger.error(f"Failed to connect to Kubernetes API port: {result}")
        except Exception as e:
            logger.error(f"Error verifying kubeconfig: {str(e)}")


    def download_helm_chart(self):
        """Download the helm chart using helm fetch"""
        logger.info("Downloading helm chart...")
        try:
            Path(self.charts_dir).mkdir(parents=True, exist_ok=True)
            
            chart_filename = self.chart_url.split('/')[-1]
            
            cmd = f"helm fetch {self.chart_url} " \
                f"--username='$oauthtoken' " \
                f"--password='{os.environ.get('NGC_API_KEY')}' " \
                f"--destination '{self.charts_dir}'"
            
            result = os.system(cmd)
            if result != 0:
                raise Exception("Helm fetch command failed")
            
            chart_path = f"{self.charts_dir}/{chart_filename}"
            logger.info(f"Successfully downloaded helm chart to {chart_path}")
            return chart_path
        except Exception as e:
            logger.error(f"Error downloading helm chart: {str(e)}")
            raise

    def setup_ssh(self):
        """
        Setup SSH configuration
        - mkdir -p ~/.ssh
        - echo "$SSH_PRIVATE_KEY" | tr -d '\r' > ~/.ssh/id_rsa
        - chmod 600 ~/.ssh/id_rsa
        - echo -e "Host *\n\tStrictHostKeyChecking no\n\n" > ~/.ssh/config
        """
        logging.info("Setting up SSH...")

        # Test network connectivity first
        self.test_network_connectivity()
        
        # Create .ssh directory
        ssh_dir = os.path.expanduser('~/.ssh')
        logging.info(f"Creating SSH directory: {ssh_dir}")
        os.makedirs(ssh_dir, mode=0o700, exist_ok=True)
        
        # Write and verify private key
        key_path = os.path.join(ssh_dir, 'id_rsa')
        logging.info(f"Writing SSH private key to: {key_path}")
        """
        Note:
        Avoid this! Directly write the key from env to file lead to trouble
        with open(key_path, 'w') as f:
            f.write(self.ssh_private_key)
        """
        # Write private key with explicit newlines
        key_lines = self.ssh_private_key.strip().split('\n')
        with open(key_path, 'w') as f:
            for line in key_lines:
                f.write(line.strip() + '\n')
        os.chmod(key_path, 0o600)
        
        # Test SSH key and connection
        self.test_ssh_connection()

    def get_kubeconfig(self):
        """
        Get kubeconfig from deployment host
        - scp byin@10.111.53.46:~/.kube/config $KUBECONFIG
        """
        logger.info("Retrieving kubeconfig from deployment host...")
        try:
            # Create kubeconfig directory if needed
            kubeconfig_dir = os.path.dirname(self.kubeconfig)
            logger.info(f"Creating kubeconfig directory: {kubeconfig_dir}")
            os.makedirs(kubeconfig_dir, exist_ok=True)
            
            """
            Note:
            - Specify the private key explicitly
            """
            # Use the same options that worked in SSH test
            scp_cmd = (
                f"scp -i ~/.ssh/id_rsa "
                f"-o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null "
                f"{self.deployment_host}:~/.kube/config {self.kubeconfig}"
            )
            logger.info(f"Executing command: {scp_cmd}")
            
            # Use subprocess for better error handling
            import subprocess
            result = subprocess.run(
                scp_cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"SCP stderr: {result.stderr}")
                raise Exception(f"SCP command failed with exit code: {result.returncode}")
            
            # Verify kubeconfig file exists and is readable
            if not os.path.isfile(self.kubeconfig):
                raise Exception(f"Kubeconfig file not found at: {self.kubeconfig}")
            
            logger.info(f"Successfully retrieved kubeconfig to: {self.kubeconfig}")
            logger.debug(f"Kubeconfig permissions: {oct(os.stat(self.kubeconfig).st_mode)[-3:]}")
        except Exception as e:
            logger.error(f"Failed to get kubeconfig: {str(e)}")
            raise

        # Verify kubeconfig after successful SCP
        self.verify_kubeconfig()


class HelmDeployer:
    def __init__(self, chart_path: str, values_file: str, kubeconfig: str, flavor: str, model_type: str):
        self.chart_path = chart_path
        self.values_file = values_file
        self.kubeconfig = kubeconfig
        self.flavor = flavor
        self.model_type = model_type
        # Get environment variables for image configuration
        self.validator_image = os.environ.get('VALIDATOR_IMAGE', None)
        self.override_image = os.environ.get('OVERRIDE_IMAGE', None)
        self.registry_image = os.environ.get('CI_REGISTRY_IMAGE', None)
        self.commit_sha = os.environ.get('CI_COMMIT_SHA', None)
        # Generate unique values file path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        values_dir = os.path.dirname(self.values_file)
        values_filename = os.path.basename(self.values_file)
        base_name, ext = os.path.splitext(values_filename)
        self.custom_values_file = os.path.join(
            values_dir,
            f"{base_name}_{self.flavor}_{self.model_type}{ext}"
        )
        self.namespace = "default"

    def get_release_name(self) -> str:
        # return f"cv-{self.flavor}-{self.model_type}"
        return "tao-cv"

    def split_docker_image(self, image_path):
        """
        Split docker image path into repository, image name, and tag.
        Image name is extracted from between last '/' and ':', with '_' replaced by '-'.

        Args:
            image_path (str): Docker image path (e.g., "nvcr.io/nim/nvidia/model_name:1.0.0")

        Returns:
            tuple: (repository, tag, image_name)
                - repository: full path without tag
                - tag: version tag or "latest"
                - image_name: model name derived from path (e.g., "model-name")
        """
        parts = image_path.rsplit(":", 1)
        repository = parts[0]
        tag = parts[1] if len(parts) > 1 else "latest"

        # Extract image name from between last '/' and ':'
        image_parts = repository.rsplit("/", 1)
        image_name = image_parts[-1].replace("_", "-")
        return repository, image_name, tag

    def update_helm_values(self):
        """Update values file with model-specific configuration and image settings"""
        logger.info(f"Creating custom values file: {self.custom_values_file}")
        try:
            with open(self.values_file, 'r') as f:
                values = yaml.safe_load(f)

            # Update main container (tao-cv)
            tao_cv_container = values['tao-cv']['applicationSpecs']['tao-cv-deployment']['containers']['tao-cv-container']
            
            # Update main container environment variables
            tao_cv_container['env'] = [
                {'name': 'NIM_MODEL_NAME', 'value': self.model_type},
                # Preserve other env vars that might exist in the values file
                *[env for env in tao_cv_container.get('env', []) 
                  if env.get('name') != 'NIM_MODEL_NAME']
            ]

            # Update validator container
            validator_container = values['tao-cv-validator']['applicationSpecs']['validator-deployment']['containers']['validator-container']
            
            # Update validator environment variables
            validator_container['env'] = [
                {'name': 'NIM_MODEL_NAME', 'value': self.model_type},  # Match main container
                # Preserve other env vars that might exist in the values file
                *[env for env in validator_container.get('env', []) 
                  if env.get('name') != 'NIM_MODEL_NAME']
            ]

            # Update image configuration
            if self.override_image:
                # Split docker image path into repository and tag
                repo, image_name, tag  = self.split_docker_image(self.override_image)
                # Validate image name matches expected format
                expected_image_name = f"nim-tao-{self.flavor}"
                if image_name != expected_image_name:
                    raise ValueError(
                        f"Invalid image name: {image_name}. "
                        f"Expected format: {expected_image_name}"
                    )
                tao_cv_container['image'] = {
                    'repository': repo,
                    'tag': tag or 'latest'
                }
            elif self.registry_image and self.commit_sha:
                # Construct image repository path matching docker push format
                image_repository = f"{self.registry_image}/nim-tao-{self.flavor}"
                logger.info(f"Setting image repository to: {image_repository}")
                
                tao_cv_container['image'] = {
                    'repository': image_repository,
                    'tag': self.commit_sha
                }
            else:
                logger.warning("Missing required environment variables for image update")
                logger.debug(f"CI_REGISTRY_IMAGE: {self.registry_image}")
                logger.debug(f"CI_COMMIT_SHA: {self.commit_sha}")
            
            # Update validator image
            if self.validator_image:
                repo, image_name, tag  = self.split_docker_image(self.validator_image)
                validator_container['image'] = {
                    'repository': repo,
                    'tag': tag
                }
            
            # Write to new values file
            os.makedirs(os.path.dirname(self.custom_values_file), exist_ok=True)
            with open(self.custom_values_file, 'w') as f:
                yaml.dump(values, f)
            
            # Log the content of the new values file
            logger.info(f"Generated custom values file content:")
            logger.info("-" * 50)
            with open(self.custom_values_file, 'r') as f:
                logger.info(f"\n{f.read()}")
            logger.info("-" * 50)
                
            logger.info("Successfully created custom values file")
            return self.custom_values_file

        except Exception as e:
            logger.error(f"Error updating helm values: {str(e)}")
            raise

    def uninstall_previous(self) -> bool:
        """Uninstall previous deployment if it exists"""
        logger.info(f"Checking for previous deployment: {self.get_release_name()}")
        
        try:
            """
            Check if release exists
            Equivalent bash command:
            helm list --filter tao-cv --kubeconfig $KUBECONFIG --output json
            """
            cmd = [
                'helm', 'list',
                '--filter', self.get_release_name(),
                '--kubeconfig', self.kubeconfig,
                '--output', 'json'
            ]
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"Helm list output: {result.stdout}")
            releases = json.loads(result.stdout if result.stdout else '[]')
            if not releases:
                logger.info("No previous deployment found")
                return True
                
            """
            Uninstall if exists
            Equivalent bash command:
            helm uninstall tao-cv --kubeconfig $KUBECONFIG --wait
            """
            logger.info(f"Uninstalling previous deployment: {self.get_release_name()}")
            cmd = [
                'helm', 'uninstall', self.get_release_name(),
                '--kubeconfig', self.kubeconfig,
                '--wait'  # Wait for resources to be deleted
            ]
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"Helm uninstall output: {result.stdout}")
            if result.stderr:
                logger.info(f"Helm uninstall stderr: {result.stderr}")
            if result.returncode != 0:
                logger.error(f"Failed to uninstall: {result.stderr}")
                return False

            """
            Wait for pods to be deleted
            Equivalent bash command:
            kubectl wait --for=delete pod -l app.kubernetes.io/instance=tao-cv --timeout=60s
            """
            logger.info("\nWaiting for pods to be deleted...")
            cmd = [
                'kubectl', '--kubeconfig', self.kubeconfig,
                'wait', '--for=delete', 'pod',
                '-n', self.namespace,
                '-l', f'app.kubernetes.io/instance={self.get_release_name()}',
                '--timeout=60s'
            ]
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"kubectl wait pods output: {result.stdout}")
            if result.stderr:
                logger.info(f"kubectl wait pods stderr: {result.stderr}")
            if result.returncode != 0:
                logger.error(f"Failed waiting for pods deletion: {result.stderr}")
                return False

            """
            Wait for services to be deleted
            Equivalent bash command:
            kubectl wait --for=delete svc -l app.kubernetes.io/instance=tao-cv --timeout=60s
            """
            logger.info("\nWaiting for services to be deleted...")
            cmd = [
                'kubectl', '--kubeconfig', self.kubeconfig,
                'wait', '--for=delete', 'svc',
                '-n', self.namespace,
                '-l', f'app.kubernetes.io/instance={self.get_release_name()}',
                '--timeout=60s'
            ]
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"kubectl wait services output: {result.stdout}")
            if result.stderr:
                logger.info(f"kubectl wait services stderr: {result.stderr}")
            if result.returncode != 0:
                logger.error(f"Failed waiting for services deletion: {result.stderr}")
                return False

            # Show final state of resources
            logger.info("\nChecking final state of resources...")
            cmd = [
                'kubectl', '--kubeconfig', self.kubeconfig,
                'get', 'all',
                '-n', self.namespace,
                '-l', f'app.kubernetes.io/instance={self.get_release_name()}'
            ]
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"Final resource state:\n{result.stdout}")
            
            logger.info(f"\n{'='*80}\nAll resources successfully deleted\n{'='*80}")
            return True
            
        except Exception as e:
            logger.error(f"Error during uninstall: {str(e)}")
            return False

    def deploy(self) -> bool:
        """Execute helm deployment"""
        try:
            # Uninstall previous deployment first
            if not self.uninstall_previous():
                logger.error("Failed to uninstall previous deployment")
                return False

            # Update values and get custom values file path
            values_file = self.update_helm_values()
            
            cmd = [
                'helm', 'upgrade', '--install',
                self.get_release_name(), self.chart_path,
                '-f', values_file,
                '--kubeconfig', self.kubeconfig,
                '--wait'
            ]
            
            logger.info(f"Executing Helm command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Helm deployment failed: {result.stderr}")
                return False
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            return False

class DeploymentValidator:
    def __init__(self, release_name: str, kubeconfig: str, namespace: str = "default"):
        self.release_name = release_name
        self.kubeconfig = kubeconfig
        self.namespace = namespace

    def check_validator_pod_status(self, timeout: int = 300) -> tuple[bool, str]:
        """
        Check if validator pod completed successfully
        Equivalent bash commands added below:
            export KUBECONFIG=/inference-builder/kubeconfig
            export NAMESPACE=default
            export RELEASE=tao-cv
        """
        logger.info("Checking validator status...")
        end_time = time.time() + timeout
        while time.time() < end_time:
            """
            Equivalent bash command:
            kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE get pods \
              -l job-name=tao-cv-validator-validator-deployment \
              -o jsonpath='{.items[0].status.phase}'
            """
            cmd = [
                'kubectl', '--kubeconfig', self.kubeconfig,
                'get', 'pods',
                '-n', self.namespace,
                '-l', f'job-name={self.release_name}-validator-validator-deployment',
                '-o', 'jsonpath={.items[0].status.phase}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to get validator pod status: {result.stderr}")
                time.sleep(10)
                continue
            
            status = result.stdout
            logger.info(f"Validator pod status: {status}")
            if status in ["Succeeded", "Completed"]:  # Accept both values
                """
                Equivalent bash command:
                VALIDATOR_POD=$(kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE get pods \
                  -l job-name=tao-cv-validator-validator-deployment \
                  -o jsonpath='{.items[0].metadata.name}')
                """
                cmd = [
                    'kubectl', '--kubeconfig', self.kubeconfig,
                    'get', 'pods',
                    '-n', self.namespace,
                    '-l', f'job-name={self.release_name}-validator-validator-deployment',
                    '-o', 'jsonpath={.items[0].metadata.name}'
                ]
                pod_result = subprocess.run(cmd, capture_output=True, text=True)
                if pod_result.returncode == 0 and pod_result.stdout:
                    pod_name = pod_result.stdout
                    """
                    Equivalent bash command:
                    kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE logs $VALIDATOR_POD
                    """
                    cmd = [
                        'kubectl', '--kubeconfig', self.kubeconfig,
                        'logs', pod_name,
                        '-n', self.namespace
                    ]
                    logs = subprocess.run(cmd, capture_output=True, text=True)
                    if "OK" in logs.stdout and "FAILED" not in logs.stdout:
                        return True, logs.stdout
                    return False, logs.stdout
                return False, "Pod completed but logs not available"
            elif status in ["Failed", "Error"]:
                # Get logs for debugging - same commands as above for pod name and logs
                cmd = [
                    'kubectl', '--kubeconfig', self.kubeconfig,
                    'get', 'pods',
                    '-n', self.namespace,
                    '-l', f'job-name={self.release_name}-validator-validator-deployment',
                    '-o', 'jsonpath={.items[0].metadata.name}'
                ]
                pod_result = subprocess.run(cmd, capture_output=True, text=True)
                if pod_result.returncode == 0 and pod_result.stdout:
                    pod_name = pod_result.stdout
                    cmd = [
                        'kubectl', '--kubeconfig', self.kubeconfig,
                        'logs', pod_name,
                        '-n', self.namespace
                    ]
                    logs = subprocess.run(cmd, capture_output=True, text=True)
                    return False, f"Validation failed: {logs.stdout}"
                return False, f"Validation failed with status: {status}"
            
            logger.info(f"Validator status: {status}, waiting...")
            time.sleep(10)
        
        return False, "Timeout waiting for validator to complete"

    def check_deployment_status(self, timeout: int = 300) -> bool:
        """Check if main service deployment is ready"""
        logger.info("Checking main deployment status...")
        end_time = time.time() + timeout
        while time.time() < end_time:
            """
            Equivalent bash command:
            kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE get deployment \
              -l app.kubernetes.io/instance=tao-cv \
              -o jsonpath='{.items[0].status.availableReplicas}'
            """
            cmd = [
                'kubectl', '--kubeconfig', self.kubeconfig,
                'get', 'deployment',
                '-n', self.namespace,
                '-l', f'app.kubernetes.io/instance={self.release_name}',
                '-o', 'jsonpath={.items[0].status.availableReplicas}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout == "1":
                logger.info("Main deployment is ready")
                return True
            logger.info("Waiting for deployment to be ready...")
            time.sleep(10)
        return False

    def validate(self) -> bool:
        """Run full validation sequence"""
        # First check main service is up
        if not self.check_deployment_status():
            logger.error("Main service deployment failed to become ready")
            return False

        # Then check validator results
        success, logs = self.check_validator_pod_status()
        if not success:
            logger.error(f"Validation failed:\n{logs}")
            return False
        
        logger.info(f"Validation succeeded:\n{logs}")
        return True

@dataclass
class StageResult:
    name: str
    success: bool
    message: str = ""
@dataclass
class ModelDeployment:
    model_id: str  # e.g., "tao/cls"
    stages: List[StageResult]

    @property
    def success(self) -> bool:
        return all(stage.success for stage in self.stages)

    def add_stage(self, name: str, success: bool, message: str = ""):
        self.stages.append(StageResult(name=name, success=success, message=message))

class DeploymentRunner:
    def __init__(self, flavor: str):
        self.flavor = flavor
        self.model_types = self._get_model_types()
        self.kubeconfig = os.path.join(os.environ['CI_PROJECT_DIR'], "kubeconfig")
        self.values_file = "builder/samples/tao/helm/tao-cv-app/custom_values.yaml"
        self.deployments: Dict[str, ModelDeployment] = {}

    def _get_model_types(self) -> List[str]:
        """Get list of model types for the flavor"""
        model_types_map = {
            "tao": ["rtdetr", "cls", "seg"],
            "changenet": ["changenet"],
            "gdino": ["gdino", "mgdino"]
        }
        return model_types_map.get(self.flavor, [])

    def record_stage_result(self, model_id: str, stage: str, success: bool, message: str = ""):
        """Record the result of a deployment stage"""
        if model_id not in self.deployments:
            self.deployments[model_id] = ModelDeployment(model_id=model_id, stages=[])
        self.deployments[model_id].add_stage(stage, success, message)
    
    def print_deployment_summary(self) -> bool:
        """Print summary of all deployments"""
        logger.info("\n" + "="*80)
        logger.info("DEPLOYMENT SUMMARY")
        logger.info("="*80 + "\n")

        for deployment in self.deployments.values():
            status_icon = "✅" if deployment.success else "❌"
            logger.info(f"{status_icon} {deployment.model_id}:")
            
            for stage in deployment.stages:
                if stage.success:
                    logger.info(f"  {stage.name}: Passed")
                else:
                    logger.error(f"  {stage.name}: Failed")
                    if stage.message:
                        logger.error(f"    → {stage.message}")
                    # Print remaining stages as skipped
                    if stage.name == "Deployment":
                        logger.info(f"  Validation: Skipped")
                    break
            logger.info("")

        success_count = sum(1 for d in self.deployments.values() if d.success)
        fail_count = len(self.deployments) - success_count

        logger.info("="*80)
        logger.info(f"Total deployments: {len(self.deployments)}")
        logger.info(f"Succeeded: {success_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info("="*80 + "\n")

        return fail_count == 0

    def _run_single_deploy(self, chart_path: str, model_type: str, teardown: bool = True) -> bool:
        """Deploy and validate a single model"""
        success = True
        model_id = f"{self.flavor}/{model_type}"
        logger.info(f"\n{'='*80}\nDeploying {model_id}\n{'='*80}")

        # Deploy stage
        deployer = HelmDeployer(
            chart_path=chart_path,
            values_file=self.values_file,
            kubeconfig=self.kubeconfig,
            flavor=self.flavor,
            model_type=model_type
        )
        deploy_success = deployer.deploy()
        self.record_stage_result(model_id, "Deployment", deploy_success)

        if deploy_success:
            # Validate stage
            release_name = deployer.get_release_name()
            validator = DeploymentValidator(
                release_name=release_name,
                kubeconfig=self.kubeconfig
            )
            validate_success = validator.validate()
            self.record_stage_result(model_id, "Validation", validate_success)
            if not validate_success:
                success = False
                logger.error(f"Validation failed for {model_id}")
            else:
                logger.info(f"Successfully validated {model_id}")
        else:
            success = False
            logger.error(f"Deployment failed for {model_id}")
        
        # Cleanup
        if teardown:
            uninstall_success = deployer.uninstall_previous()
            if not uninstall_success:
                logger.error(f"Failed to cleanup deployment for {model_id}")

        return success

    def run_all(self, chart_path: str) -> bool:
        """Run deployment for all model types in the flavor"""
        self.model_types = self._get_model_types()
        success = True

        # for each model type, deploy -> validate -> teardown
        for model_type in self.model_types:
            success = self._run_single_deploy(chart_path=chart_path, model_type=model_type, teardown=True)
            if not success:
                logger.error(f"Deployment failed for {model_type}")
                success = False

        return success

    def run_user_deploy(self, chart_path: str, model_type: str) -> bool:
        """Run single model deployment for user mode"""
        # Verify model type is valid for flavor
        if model_type not in self._get_model_types():
            logger.error(f"Invalid model type '{model_type}' for flavor '{self.flavor}'. "
                        f"Valid types are: {self._get_model_types()}")
            return False
        success = self._run_single_deploy(chart_path=chart_path, model_type=model_type, teardown=False)

        return success
    
    def parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description='Deploy and validate TAO CV models')
        parser.add_argument('--mode', choices=['test', 'deploy'], default='deploy',
                          help='Run mode: test (test all models and cleanup) or deploy (single model, keep running)')
        
        # Only require flavor and model-type for deploy mode
        parser.add_argument('--flavor', type=str, 
                          help='Model flavor (tao, changenet, gdino)')
        parser.add_argument('--model-type', type=str, 
                          help='Specific model type to deploy (e.g., rtdetr, cls, seg)')
        
        args = parser.parse_args()

        # Validate arguments based on mode
        if args.mode == 'test':
            # test mode: get flavor from env
            self.flavor = os.environ.get('FLAVOR')
            if not self.flavor:
                parser.error("FLAVOR environment variable must be set for test mode")
        else:
            # Deploy mode: require flavor and model-type args
            if not args.flavor:
                parser.error("--flavor is required for deploy mode")
            if not args.model_type:
                parser.error("--model-type is required for deploy mode")
            self.flavor = args.flavor
        return args

    def run(self) -> bool:
        """Main entry point for the deployment runner"""
        args = self.parse_args()

        # One-time setup
        prep = DeploymentPreparation()
        chart_path = prep.prepare()
        
        success = False
        if args.mode == 'test':
            success = self.run_all(chart_path=chart_path)
        else:  # deploy mode
            success = self.run_user_deploy(chart_path=chart_path, model_type=args.model_type)

        self.print_deployment_summary()
        return success


if __name__ == '__main__':
    runner = DeploymentRunner("")  # Empty flavor initially, will be set by parse_args
    if not runner.run():
        raise Exception("Deployment failed")
