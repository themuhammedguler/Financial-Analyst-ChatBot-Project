"""Hugging Face Spaces Senkronizasyon Betiği.

GitHub `README.md` dosyasını temiz tutar (frontmatter tablosu olmadan),
Hugging Face Spaces'e push ederken ise gerekli YAML frontmatter bloğunu
otomatik ekleyerek Spaces uygulamasının kesintisiz çalışmasını sağlar.
"""

import subprocess
import sys


def sync():
    try:
        tree_sha = subprocess.check_output(["git", "rev-parse", "main^{tree}"]).decode().strip()
        readme_content = subprocess.check_output(["git", "show", "main:README.md"]).decode("utf-8")

        frontmatter = (
            "---\n"
            "title: BIST Finansal Analist Chatbot\n"
            "emoji: 📈\n"
            "colorFrom: blue\n"
            "colorTo: indigo\n"
            "sdk: streamlit\n"
            "sdk_version: 1.40.1\n"
            "app_file: app.py\n"
            "pinned: false\n"
            "license: mit\n"
            "---\n\n"
        )
        hf_readme = frontmatter + readme_content

        p = subprocess.Popen(
            ["git", "hash-object", "-w", "--stdin"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        blob_sha, _ = p.communicate(input=hf_readme.encode("utf-8"))
        blob_sha = blob_sha.decode().strip()

        ls_tree = subprocess.check_output(["git", "ls-tree", tree_sha]).decode("utf-8")
        new_ls_lines = []
        for line in ls_tree.strip().split("\n"):
            if line.endswith("\tREADME.md"):
                mode = line.split()[0]
                new_ls_lines.append(f"{mode} blob {blob_sha}\tREADME.md")
            else:
                new_ls_lines.append(line)

        new_ls_data = "\n".join(new_ls_lines) + "\n"

        p = subprocess.Popen(["git", "mktree"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        new_tree_sha, _ = p.communicate(input=new_ls_data.encode("utf-8"))
        new_tree_sha = new_tree_sha.decode().strip()

        # Space remote'un HEAD sha'sını al veya ilk push ise doğrudan yeni kök commit yap
        try:
            space_head = (
                subprocess.check_output(["git", "ls-remote", "space", "refs/heads/main"])
                .decode()
                .split()[0]
            )
            parents = ["-p", space_head]
        except Exception:
            parents = []

        commit_cmd = ["git", "commit-tree", new_tree_sha] + parents + ["-m", "deploy: sync main with HF space frontmatter"]
        commit_sha = subprocess.check_output(commit_cmd).decode().strip()

        subprocess.check_call(["git", "push", "space", f"{commit_sha}:refs/heads/main", "--force"])
        print("✅ Hugging Face Spaces ile başarıyla senkronize edildi!")
    except Exception as e:
        print(f"❌ Hata oluştu: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sync()
