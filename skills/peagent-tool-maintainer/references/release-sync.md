# Release Sync Policy

## Direct-To-Main Flow

After a feature or fix is confirmed:

1. Check repository state:

   ```bash
   skills/peagent-tool-maintainer/scripts/check_ready.py --repo /scratch/jy16611/peagent --allow-dirty
   git status --short
   ```

2. Run tests and docs checks relevant to the change.
3. Stage only current-task files.
4. Inspect staged changes:

   ```bash
   git diff --cached --stat
   git diff --cached --name-only
   ```

5. Run `check_ready.py` again to catch staged large or forbidden files.
6. Commit to `main` with a clear imperative message.
7. Push:

   ```bash
   git push origin main
   ```

## Stop Conditions

Do not commit or push when:

- unrelated dirty files are present and cannot be separated safely;
- tests, compile checks, or docs builds fail;
- staged files include model weights, runtime metadata, data dumps, envs, caches, or files larger than the configured limit;
- `git push` is rejected or remote history differs from local expectations.

Report the exact blocker and the command that failed.
