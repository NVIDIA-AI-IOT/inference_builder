# nim-templates

## Getting started

### Clone the repository

```bash
git clone https://gitlab-master.nvidia.com/chunlinl/nim-templates.git
git submodule update --init
```

### Install prerequisites

```bash
sudo apt update
sudo apt install protobuf-compiler
```

Create python virtual env (Optional) and install dependent packages

```bash
$ python -m venv /path/to/new/virtual/environment
# Activate the venc
$ source /path/to/new/virtual/environment/bin/activate
# Install the dependent packages
$ cd nim-templates
$ pip3 install -r requirements.txt

```

## NIM Dependencies

Ensure nvidia runtime added to `/etc/docker/daemon.json`

```bash
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

Install Nim Tools

```bash
$ pip install nimtools==0.4.0 --index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple 

#Check installation
$ nim_builder --version 
nim_builder 0.4.0 nimlib 0.1.47 nim-compliance 2.0.0
```

## Usage
The project provides developers an easy-to-use command line tool to generate inference codes for various VLM/CV NIMs. Before running the tool, developers need to create a comprehensive inference configuration file and the API spec. For some NIMs, custom code snippet would also be needed.

Under the tests/configs we provides a sample configuration for creating vila NIM, the inference code can be generate from the following command:

```bash
pyton builder/main.py
usage: Inference Builder [-h] [--server-type [{triton}]] [-o [OUTPUT_DIR]] [-a [API_SPEC]] [-c [CUSTOM_MODULE ...]] [-x] [-t] config
```

There're two builtin samples under _samples_ folder for generating vila NIMs and nvclip NIMs respectively.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
