HELM_CMD ?= microk8s helm3
MS_CHARTS:=$(wildcard ../services/*/output/helm/Chart.yaml)

all: build

build: build_all_ms
build: output/Chart.yaml output/templates/secrets.yaml

build_all_ms:
	$(MAKE) -C ../services

output/Chart.yaml: $(MS_CHARTS) tao-cv-app.yaml
	ucf_app_builder_cli app build tao-cv-app.yaml -f -o output/

output/templates/secrets.yaml:
	#@./add_secrets.sh
	echo "Not updating secrets"

install_or_upgrade:
	@$(HELM_CMD) ls | grep tao-cv-app >/dev/null && $(MAKE) upgrade || $(MAKE) install


install: INSTALL_CMD=install
	export INSTALL_CMD

install: build output/.deployed

upgrade: INSTALL_CMD=upgrade
	export INSTALL_CMD

upgrade: build output/.deployed

# The .deployed file acts as a marker that Make uses to know if/when the deployment was last done.
output/.deployed: output/Chart.yaml
	$(HELM_CMD) $(INSTALL_CMD) tao-cv-app output -f custom_values.yaml
	@touch output/.deployed

clean:
	rm -rf output
	$(MAKE) -C ../services clean


uninstall:
	@$(HELM_CMD) uninstall tao-cv-app
	@rm -f output/.deployed

