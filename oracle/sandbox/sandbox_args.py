"""Wrap a command in the Mac sandbox used for model-written code: no network, writes to temp only, no reading of secrets."""
import os, pathlib
HERE = pathlib.Path(__file__).parent.resolve()
HOME = os.path.expanduser("~")
def wrap(cmd):
    return ["sandbox-exec", "-f", str(HERE / "sandbox.sb"), "-D", f"HOME_SSH={HOME}/.ssh", "-D", f"HOME_AWS={HOME}/.aws", "-D", f"HOME_MODAL={HOME}/.modal",
            "-D", f"HOME_MODAL_TOML={HOME}/.modal.toml", "-D", f"ENV_FILE={HERE.parent.parent}/.env", *cmd]
