"""Package tracked and unignored source files, normalizing text to LF for Linux."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
    contents = {}
    for name in sorted(set(names)-{""}):
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if path.suffix in (".py", ".sh", ".slurm", ".md", ".txt", ".json", ".toml", ".cpp", ".h", ".ps1", ".js", ".css", ".html"):
            data = data.replace(b"\r\n", b"\n")
        contents[name] = data
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    version = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()
    manifest = {"source_version": version, "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(), "files": checksums}
    contents["SOURCE.json"] = json.dumps(manifest, indent=2).encode()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(output, "w:gz") as archive:
        for name, data in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    print(json.dumps({"path": str(output.resolve()), "source_version": version,
                      "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
