"""Publish the prepared Adivyanta package to the authenticated user's Hub account."""
import argparse

from huggingface_hub import HfApi, get_token

from export_huggingface import PACKAGE, main as export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", help="Target model repo, default: <logged-in-user>/Adivyanta-46M")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    if not get_token():
        raise SystemExit("Hugging Face login required. Run .\\.venv\\Scripts\\hf.exe auth login first.")
    api = HfApi()
    username = api.whoami()["name"]
    repo_id = args.repo_id or f"{username}/Adivyanta-46M"
    if not args.skip_export:
        export()
    if not (PACKAGE / "model.safetensors").exists():
        raise FileNotFoundError(PACKAGE / "model.safetensors")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    result = api.upload_folder(
        folder_path=str(PACKAGE), repo_id=repo_id, repo_type="model",
        commit_message="Release Adivyanta 46M scratch checkpoints, model card and benchmarks",
    )
    print(f"Uploaded https://huggingface.co/{repo_id}")
    print(f"Commit: {result}")


if __name__ == "__main__":
    main()
