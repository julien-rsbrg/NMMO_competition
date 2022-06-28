#!/bin/sh
''''exec python -u "$0" ${1+"$@"} # '''

import os
import sys
import time
import json
import traceback
import termcolor
import threading
import subprocess
import multiprocessing as mp
from subprocess import CalledProcessError
from pathlib import Path

from ijcai2022nmmo import submission as subm

IMAGE = "ijcai2022nmmo/submission-runtime"
CONTAINER = "ijcai2022-nmmo-runner"
PORT = 12343
TENCENTCLOUD_REGISTRY = "ccr.ccs.tencentyun.com"
MAX_REPO_SIZE = 500


def run_team_server(submission_path: str):
    subm.check(submission_path)
    team_klass, init_params = subm.parse_submission(submission_path)
    print(f"Start TeamServer for {team_klass.__name__}")
    from ijcai2022nmmo import TeamServer
    server = TeamServer("0.0.0.0", PORT, team_klass, init_params)
    server.run()


def run_submission_in_process(submission_path) -> mp.Process:
    mp.set_start_method("fork", force=True)

    def run_server():
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        run_team_server(submission_path)

    p = mp.Process(target=run_server, daemon=True)
    p.start()
    return p


def run_submission_in_docker(submission_path, registry):
    if registry not in ["dockerhub", "tencentcloud"]:
        err(f"Invalid registry {registry}")
        sys.exit(6)
    ok(f"Use docker registry [{registry}]")

    need_root = True if os.system("docker ps 1>/dev/null 2>&1") else False

    def _shell(command, capture_output=True, print_command=True):
        if command.startswith("docker") and need_root:
            command = "sudo " + command
        if print_command:
            print(command)
        r = subprocess.run(command, shell=True, capture_output=capture_output)
        if not capture_output: return r.returncode
        if r.returncode != 0:
            # grep return 1 when no lines matching
            if r.returncode == 1 and "grep" in command:
                pass
            else:
                raise CalledProcessError(r.returncode, r.args, r.stdout,
                                         r.stderr)
        return r.stdout.decode().strip()

    # check if should pull manually
    manual_pull = False
    if registry != "dockerhub":
        with open("Dockerfile", "r") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith(
                        "FROM") and line.split()[-1] == f"{IMAGE}:latest":
                    manual_pull = True
                    break

    if manual_pull:
        if registry == "tencentcloud":
            ok(f"Try pull image from {TENCENTCLOUD_REGISTRY}")
            if _shell(f"docker pull {TENCENTCLOUD_REGISTRY}/{IMAGE}:latest",
                      capture_output=False):
                err(f"Pull image failed.")
                sys.exit(10)
            _shell(
                f"docker tag {TENCENTCLOUD_REGISTRY}/{IMAGE}:latest {IMAGE}:latest",
                capture_output=False)
            manual_pull = True
        else:
            assert 0, f"Invalid registry {registry}"

    ok(f"Try build image {IMAGE}:local ...")
    if manual_pull:
        if _shell(f"docker build -t {IMAGE}:local -f Dockerfile .",
                  capture_output=False):
            err("Build failed.")
            sys.exit(10)
    else:
        if _shell(f"docker build --pull -t {IMAGE}:local -f Dockerfile .",
                  capture_output=False):
            err("Build failed.")
            sys.exit(10)
    if _shell(f'docker ps -a | grep -w "{CONTAINER}"') != "":
        _shell(f"docker stop {CONTAINER}")
        _shell(f"docker rm {CONTAINER}")
    # cwd = Path(__file__).parent.resolve().as_posix()
    command = f"python tool.py run_team_server --submission={submission_path}"
    container_id = _shell(
        f"docker run -it -d --name {CONTAINER} -p {PORT}:{PORT} {IMAGE}:local {command}"
    )

    threading.Thread(target=_shell,
                     args=(f"docker logs -f {container_id}", False),
                     daemon=True).start()

    def _check_container_alive():
        while 1:
            ret = _shell(
                f"docker inspect {container_id} --format='{{{{.State.ExitCode}}}}'",
                print_command=False)
            if ret != "0":
                err(f"Container {container_id} exit unexpectedly")
                os._exit(1)
            time.sleep(1)

    threading.Thread(target=_check_container_alive, daemon=True).start()

    return container_id


def ok(msg: str):
    print(termcolor.colored(msg, "green", attrs=['bold']))


def warn(msg: str):
    print(termcolor.colored(msg, "yellow", attrs=['bold']))


def err(msg: str):
    print(termcolor.colored(msg, "red", attrs=['bold']))


def rollout(submission_path: str, startby: str, registry: str):
    from ijcai2022nmmo import CompetitionConfig

    class Config(CompetitionConfig):
        NMAPS = 1

    if startby == "docker":
        ok(f"Try run submission in docker container ...")
        container_id = run_submission_in_docker(submission_path, registry)
        ok(f"Submission is running in container {container_id}")
    elif startby == "process":
        ok(f"Try run submission in subprocess ...")
        p = run_submission_in_process(submission_path)
        ok(f"Submission is running in process {p.pid}")
    else:
        err(f"startby should be either docker or process")
        sys.exit(1)

    from ijcai2022nmmo import ProxyTeam
    team = ProxyTeam("my-submission", Config(), "127.0.0.1", PORT)

    try:
        from ijcai2022nmmo import RollOut, scripted
        ro = RollOut(
            Config(),
            [
                scripted.RandomTeam(f"random-{i}", Config())
                for i in range(CompetitionConfig.NPOP - 1)
            ] + [team],
            True,
        )
        ro.run()
    except:
        raise
    finally:
        team.stop()


def check_repo_size() -> bool:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        import platform
        system = platform.system()
        if system == "Darwin":
            r = subprocess.run(f"du -sm -I .git {current_dir}",
                               shell=True,
                               capture_output=True)
        else:
            r = subprocess.run(f"du -sm --exclude=.git {current_dir}",
                               shell=True,
                               capture_output=True)
        out = r.stdout.decode().strip()
        size = int(out.split()[0])
    except Exception as e:
        warn(f"Get repo size failed -- {e}")
        return True
    if size > MAX_REPO_SIZE:
        warn(f"Current repo size: {size}MB")
        err(f"Please try making your repo size (.git doesn't count) smaller than {MAX_REPO_SIZE}MB."
            )
        return False
    else:
        ok(f"Current repo size: {size}MB")
    return True


class Toolkit:
    def test(
        self,
        startby: str = "process",
        registry: str = "dockerhub",
    ):
        r"""
        Tests your submission in ``my-submission/`` by running rollout.

        Args:
            startby (Optional[str], optional): Defines how the submission code will be running. Options: ["docker", "process]
            registry (str, optional): Defines where to pull the docker image. It has effect only when ``startby=docker``. Options: ["dockerhub", "tencentcloud"]
        """
        assert isinstance(startby, str)
        assert isinstance(registry, str)

        submission = "my-submission"
        try:
            subm.check(submission)
        except Exception as e:
            traceback.print_exc()
            err(str(e))
            sys.exit(5)
        ok(f"Testing {submission} ...")

        from art import text2art
        try:
            rollout(submission, startby, registry)
        except:
            traceback.print_exc()
            err(text2art("TEST FAIL", "sub-zero"))
            sys.exit(1)
        else:
            ok(text2art("TEST PASS", "sub-zero"))

    def run_team_server(self, submission: str):
        run_team_server(submission)

    def check_aicrowd_json(self):
        with open("aicrowd.json", "r") as fp:
            config: dict = json.load(fp)

        if config.get("challenge_id") != "ijcai-2022-the-neural-mmo-challenge":
            err(f"[challenge_id] in aicrowd.json should be ijcai-2022-the-neural-mmo-challenge"
                )
            sys.exit(3)

        if not config.get("authors"):
            authors = []
            warn(f"[authors] in aicrowd.json is empty")
            ok(f"Enter the authors (seperated by comma(,)).")
        else:
            authors = config["authors"]
            ok(f"Current authors are: {config['authors']}")
            ok(f"Enter the authors (seperated by comma(,)). If no change to the authors, just press ENTER."
               )

        text = input(": ").strip()
        if text:
            authors = [
                x.strip().replace("'", "").replace('"', '')
                for x in text.split(",") if x.strip()
            ]
        if not authors:
            err(f"authors are empty")
            sys.exit(4)

        ok(f"Current authors are: {authors}")

        config["authors"] = authors
        with open("aicrowd.json", "w") as fp:
            fp.write(json.dumps(config, indent=4))

    def aicrowd_setup(self):
        r"""
        Setting up the AICrowd config.
        """
        from aicrowd.contexts.config import CLIConfig
        c = CLIConfig()
        c.load(None)
        if not c.get("aicrowd_api_key"):
            warn("AICrowd config not found. Trying ``aicrowd login``.")
            r = subprocess.run("aicrowd login",
                               shell=True,
                               capture_output=False)
            if r.returncode:
                err("aicrowd_setup failed.")
        ok("aicrowd_setup done.")

    def submit(
        self,
        submission_id,
        skip_test: bool = False,
        startby: str = "process",
        registry: str = "dockerhub",
    ):
        r"""
        Pushes your submission.

        Args:
            submission_id (str): Name of submission. Each submission must have a unique name.
            skip_test (bool, optional): If it is set, there will be no test before pushing.
            startby (Optional[str], optional): Defines how the submission code will be running. Options: ["docker", "process]
            registry (str, optional): Defines where to pull the docker image. It has effect only when ``startby=docker``. Options: ["dockerhub", "tencentcloud"]
        """
        assert isinstance(skip_test, bool)
        assert startby is None or isinstance(startby, str)
        assert isinstance(registry, str)

        warn(f"Please make sure putting all your submission related (code, model, ...) in the my-submission folder.")

        if not check_repo_size():
            err("check_repo_size failed.")
            sys.exit(1)

        self.aicrowd_setup()

        self.check_aicrowd_json()

        if not skip_test:
            self.test(startby, registry)

        r = subprocess.run(f"bash .submit.sh '{submission_id}'",
                           shell=True,
                           capture_output=False)
        if r.returncode:
            err("bash .submit.sh failed.")
            sys.exit(1)


if __name__ == "__main__":
    import fire
    fire.Fire(Toolkit)
