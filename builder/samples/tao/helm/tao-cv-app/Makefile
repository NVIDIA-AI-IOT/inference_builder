HELM_CMD ?= microk8s helm3
KUBECTL_CMD ?= microk8s kubectl
MS_CHARTS:=$(wildcard ../services/*/output/helm/Chart.yaml)

# Add these variables at the top with other variables
NGC_ORG ?= eevaigoeixww
NGC_TEAM ?= dev
VERSION ?=

all: build

build: build_all_ms

build_all_ms:
	@echo "Building all dependent microservices..."
	@echo
	$(MAKE) -C ../services
	@echo "All microservices built successfully"

output/Chart.yaml: $(MS_CHARTS) tao-cv-app.yaml
	@echo "Generating App helm chart..."
	@echo
	ucf_app_builder_cli app build tao-cv-app.yaml -f -o output/
	@echo "Helm chart generated successfully"


install_or_upgrade:
	@echo "Checking if tao-cv-app exists..."
	@echo
	@$(HELM_CMD) ls | grep tao-cv-app >/dev/null && $(MAKE) upgrade || $(MAKE) install


install: INSTALL_CMD=install
	export INSTALL_CMD

install: build output/.deployed

upgrade: INSTALL_CMD=upgrade
	export INSTALL_CMD

upgrade: build output/.deployed

# The .deployed file acts as a marker that Make uses to know if/when the deployment was last done.
output/.deployed: output/Chart.yaml
	@echo "Deploying tao-cv-app with $(INSTALL_CMD)..."
	@echo
	$(HELM_CMD) $(INSTALL_CMD) tao-cv-app output -f custom_values.yaml
	@echo
	@echo "Applying NodePort service..."
	$(KUBECTL_CMD) apply -f ../services/tao-cv/nodeport.yaml
	@touch output/.deployed

clean:
	rm -rf output
	$(MAKE) -C ../services clean


status:
	@echo "Checking tao-cv-app deployment status..."
	@echo
	$(HELM_CMD) list | grep tao-cv-app
	@echo
	@echo "Helm-managed resources:"
	$(KUBECTL_CMD) get all -l "app.kubernetes.io/instance=tao-cv-app"
	@echo
	@echo "NodePort service status:"
	$(KUBECTL_CMD) get service tao-cv-nodeport

uninstall:
	@echo "Uninstalling tao-cv-app..."
	@echo
	$(HELM_CMD) uninstall tao-cv-app
	@echo
	@echo "Removing NodePort service..."
	$(KUBECTL_CMD) delete -f ../services/tao-cv/nodeport.yaml
	@rm -f output/.deployed
	@echo "Uninstall completed"

package:
	@echo "Packaging tao-cv-app..."
	@echo
	cd output && $(HELM_CMD) package .
	@echo "Packaging completed"

push_ngc:
	@if [ -z "$(NGC_ORG)" ] || [ -z "$(NGC_TEAM)" ] || [ -z "$(VERSION)" ]; then \
		echo "Error: Missing required parameters. Usage:"; \
		echo "make push_ngc NGC_ORG=<org> NGC_TEAM=<team> VERSION=<version>"; \
		exit 1; \
	fi
	@echo "Pushing tao-cv-app to NGC..."
	@echo
	cd output && \
	ngc registry chart push $(NGC_ORG)/$(NGC_TEAM)/tao-cv-app:$(VERSION)
	@echo "Pushing completed"